import torch
import os
import subprocess
import tensorrt as trt
import sys
import atexit
import ctypes
import modelopt.torch.quantization as mtq
import copy
import gc
from typing import Dict, List, Tuple
import shutil

import numpy as np
import torch


FP8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

NVFP4_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

    

def wan_quantize(
    policy,
    quantization_config,
    model_type,
    forward_loop,
):
    """Quantize the VLA model using ModelOpt - simplified to use calc_mse_for_single_trajectory."""

    # Configure quantization - disable problematic layers.
    # ModelOpt 0.44 uses a list-of-dicts quant_cfg format; older configs used a
    # mapping. Support both so local custom configs remain usable.
    if "quant_cfg" in quantization_config:
        quant_cfg = quantization_config["quant_cfg"]
        if isinstance(quant_cfg, list):
            quant_cfg.append({"quantizer_name": "*patch_embedding*", "enable": False})
        else:
            quant_cfg["*patch_embedding*"] = {"enable": False}
    #    if model_type == "14B" or model_type == "ar_14B":
    #        # Workaround: until we understand the issue https://nvbugspro.nvidia.com/bug/5612316
    #        quantization_config["quant_cfg"]["*.self_attn.o.*"] = {"enable": False}
    #        quantization_config["quant_cfg"]["*.cross_attn.o.*"] = {"enable": False}

    policy.trained_model.action_head.model = mtq.quantize(
        policy.trained_model.action_head.model, quantization_config, forward_loop=forward_loop
    )
    if os.getenv("PRINT_QUANT_SUMMARY", "false").lower() == "true":
        mtq.print_quant_summary(policy.trained_model.action_head.model)

    return


def wan_trt_quantize_and_load_engine(
    policy,
    cfg,
    onnx_path,
    engine_path,
    model_type,
    forward_loop,
):
    if cfg.inference_mode == "trt_build":
        for path in (engine_path, onnx_path):
            if os.path.exists(path):
                os.remove(path)

    quantization_config = None
    if cfg.quantize_dtype == "fp8":
        quantization_config = copy.deepcopy(getattr(mtq, "FP8_DEFAULT_CFG", FP8_DEFAULT_CONFIG))
    elif cfg.quantize_dtype == "nvfp4":
        quantization_config = copy.deepcopy(getattr(mtq, "NVFP4_DEFAULT_CFG", NVFP4_DEFAULT_CONFIG))
    else:
        print(f"Quantization type {cfg.quantize_dtype} not supported. Skipping quantization.")

    if quantization_config is not None and cfg.inference_mode == "trt_build":
        wan_quantize(
            policy,
            quantization_config,
            model_type=model_type,
            forward_loop=forward_loop,
        )

    if  cfg.inference_mode == "trt_build":
        # Cast the DiT to fp16 for ONNX/TensorRT export AFTER calibration. The
        # wan_quantize calibration above runs a REAL-data forward whose activations
        # are bf16 (VAE/text/image encoders stay bf16), so the DiT must also be bf16
        # during calibration. The ONNX export below uses fp16 test inputs
        # (create_wan_test_inputs -> torch.float16), so the model is cast to fp16
        # only here. NOTE: a prior change moved this cast BEFORE wan_quantize, which
        # fed bf16 activations into an fp16 DiT and crashed calibration with
        # "mat1 and mat2 must have the same dtype, but got BFloat16 and Half".
        policy.trained_model.action_head.model.to(torch.float16)

        if os.getenv("PRINT_TRT_EXPORT_MODEL", "false").lower() == "true":
            print("Export model:", policy.trained_model.action_head.model)

        test_inputs = create_wan_test_inputs(policy, device="cuda", model_type=model_type)
        min_shape = None
        max_shape = None
        opt_shape = None

        if model_type == "ar_14B":

            policy.trained_model.action_head.model.forward = policy.trained_model.action_head.model._forward_inference_trt
            dynamic_axes = {
                "kv_cache_packed": {3: "kv_cache_len"},
            }
            min_shape = "kv_cache_packed:40x2x1x880x40x128"
            max_shape = "kv_cache_packed:40x2x1x8800x40x128"
            opt_shape = "kv_cache_packed:40x2x1x7920x40x128"
        elif model_type == "ar_14B_droid":
            policy.trained_model.action_head.model.forward = policy.trained_model.action_head.model._forward_inference_trt
            dynamic_axes = {
                "kv_cache_packed": {3: "kv_cache_len"},
            }
            min_shape = "kv_cache_packed:40x2x1x880x40x128"
            max_shape = "kv_cache_packed:40x2x1x8800x40x128"
            opt_shape = "kv_cache_packed:40x2x1x7920x40x128"
        elif model_type == "ar_5B_n6":
            policy.trained_model.action_head.model.forward = policy.trained_model.action_head.model._forward_inference_trt
            dynamic_axes = {
                "kv_cache_packed": {3: "kv_cache_len"},
            }
            min_shape = "kv_cache_packed:30x2x1x220x24x128"
            max_shape = "kv_cache_packed:30x2x1x3080x24x128"
            opt_shape = "kv_cache_packed:30x2x1x2860x24x128"
        else:
            dynamic_axes = None

        if cfg.quantize_dtype == "nvfp4":
            exported_onnx_path = export_to_onnx_fp4(policy.trained_model.action_head.model, test_inputs, onnx_path, dynamic_axes=dynamic_axes)
        else:
            exported_onnx_path = export_to_onnx(
                policy.trained_model.action_head.model,
                test_inputs,
                onnx_path,
                model_type=model_type,
                quantization_mode=cfg.quantize_dtype,
                dynamic_axes=dynamic_axes,
            )
        if exported_onnx_path is None or not os.path.exists(onnx_path):
            raise RuntimeError(f"ONNX export failed; expected ONNX file at {onnx_path}")
        if cfg.quantize_dtype == "nvfp4":
            _sanitize_nvfp4_onnx_for_tensorrt(onnx_path)

        # Free the large PyTorch model before TensorRT allocates builder memory.
        # The build script only needs the serialized engine after export.
        policy.trained_model.action_head.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        built_engine_path = build_tensorrt_engine(onnx_path, engine_path, min_shape, max_shape, opt_shape)
        if built_engine_path is None or not os.path.exists(engine_path):
            raise RuntimeError(f"TensorRT engine build failed; expected engine at {engine_path}")

    trt_wan_model = load_tensorrt_engine(engine_path, model_type=model_type)
    policy.trained_model.action_head.model = trt_wan_model

def export_to_onnx_fp4(model, test_inputs, onnx_save_path, dynamic_axes=None):
    from modelopt.torch._deploy.utils.torch_onnx import OnnxBytes
    from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata

    print("exporting to onnx fp4")
    try:
        onnx_bytes, _ = get_onnx_bytes_and_metadata(model=model, dummy_input=test_inputs, dynamic_axes=dynamic_axes)
        onnx_model = OnnxBytes.from_bytes(onnx_bytes)
    except Exception as e:
        import traceback as _tb
        print(f"Error exporting model to ONNX: {e!r}")
        _tb.print_exc()
        raise
    save_dir = os.path.dirname(os.path.abspath(onnx_save_path))
    os.makedirs(save_dir, exist_ok=True)
    for filename, file_bytes in onnx_model.onnx_model.items():
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        print(f"exported onnx to {file_path}")
    return onnx_save_path


def _sanitize_nvfp4_onnx_for_tensorrt(onnx_path: str) -> None:
    """Patch ModelOpt NVFP4 ONNX graphs for TensorRT's strict MatMul typing.

    ModelOpt 0.44 can emit FP4 weight DequantizeLinear nodes whose output is
    inferred as FP32 because the scale tensor is FP32. The matching activation
    side is FP16, so TensorRT rejects MatMul with Half x Float inputs. Cast only
    the FP4-weight side to FP16 before MatMul; the compressed FP4 initializer is
    kept intact.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(onnx_path, load_external_data=False)
    producers = {output: node for node in model.graph.node for output in node.output}
    new_nodes = []
    cast_outputs: dict[tuple[str, str], str] = {}
    matmul_patched = 0
    layernorm_patched = 0

    def is_fp4_weight_transpose(value_name: str) -> bool:
        transpose = producers.get(value_name)
        if transpose is None or transpose.op_type != "Transpose" or not transpose.input:
            return False
        dq = producers.get(transpose.input[0])
        return (
            dq is not None
            and dq.op_type == "DequantizeLinear"
            and bool(dq.input)
            and dq.input[0].endswith("_f4")
        )

    def static_tensor_dtype(value_name: str) -> int | None:
        for init in model.graph.initializer:
            if init.name == value_name:
                return init.data_type
        producer = producers.get(value_name)
        if producer is None or producer.op_type != "Constant":
            return None
        for attr in producer.attribute:
            if attr.name == "value":
                return attr.t.data_type
        return None

    def cast_to_fp16(value_name: str, node_name: str, reason: str) -> str:
        key = (value_name, reason)
        cast_output = cast_outputs.get(key)
        if cast_output is not None:
            return cast_output
        cast_output = f"{value_name}_{reason}_fp16"
        cast_node = helper.make_node(
            "Cast",
            inputs=[value_name],
            outputs=[cast_output],
            name=f"{node_name or 'node'}_{reason}_cast_fp16",
            to=TensorProto.FLOAT16,
        )
        new_nodes.append(cast_node)
        cast_outputs[key] = cast_output
        return cast_output

    for node in model.graph.node:
        if node.op_type == "MatMul":
            for input_idx, input_name in enumerate(list(node.input)):
                if not is_fp4_weight_transpose(input_name):
                    continue
                node.input[input_idx] = cast_to_fp16(input_name, node.name or "MatMul", "fp4_weight")
                matmul_patched += 1
        elif node.op_type == "LayerNormalization" and len(node.input) >= 2:
            if (
                static_tensor_dtype(node.input[1]) == TensorProto.FLOAT16
                and not node.input[0].endswith("_layernorm_input_fp16")
            ):
                node.input[0] = cast_to_fp16(node.input[0], node.name or "LayerNormalization", "layernorm_input")
                layernorm_patched += 1
        new_nodes.append(node)

    if matmul_patched == 0 and layernorm_patched == 0:
        print("  NVFP4 ONNX sanitization: no TensorRT MatMul type patches needed")
        return

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save(model, onnx_path)
    print(
        "  NVFP4 ONNX sanitization: inserted "
        f"{len(cast_outputs)} FP16 casts "
        f"({matmul_patched} MatMul inputs, {layernorm_patched} LayerNorm inputs)"
    )


def export_to_onnx(
    pytorch_model,
    test_inputs,
    onnx_path="tensorrt/wan_model.onnx",
    model_type="5B",
    quantization_mode="fp8",
    dynamic_axes=None,
):
    #
    if model_type == "5B":
        return export_to_onnx_5B(pytorch_model, test_inputs, onnx_path, dynamic_axes)
    elif model_type == "14B":
        return export_to_onnx_14B(pytorch_model, test_inputs, onnx_path, dynamic_axes)
    elif model_type == "ar_14B" or model_type == "ar_14B_droid":
        return export_to_onnx_ar_14B(pytorch_model, test_inputs, onnx_path, dynamic_axes)
    else:
        raise ValueError(f"Model type {model_type} not supported")


def export_to_onnx_ar_14B(pytorch_model, test_inputs, onnx_path="tensorrt/wan_model.onnx", dynamic_axes=None):
    """Export PyTorch model to ONNX"""
    print("Exporting AR 14B model to ONNX...", onnx_path)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pytorch_model.eval()
    pytorch_model.to(torch.float16)

    input_names = [
        "x",
        "timestep",
        "context",
        "kv_cache_packed",
        "y",
        "clip_feature",
        "action",
        "timestep_action",
        "state",
    ]
    output_names = ["video_noise_pred", "action_noise_pred"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                test_inputs,
                onnx_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        print(f"  ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        import traceback
        print(f"  ERROR: ONNX export failed. Exception type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None



def export_to_onnx_5B(pytorch_model, test_inputs, onnx_path="tensorrt/wan_model.onnx"):
    """Export PyTorch model to ONNX"""
    print("Exporting model to ONNX...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pytorch_model.eval()
    pytorch_model.to(torch.float16)

    x, action, timestep, context, state, embodiment_id = test_inputs

    # Define input names for better ONNX graph
    input_names = ["x", "action", "timestep", "context", "state", "embodiment_id"]
    output_names = ["video_noise_pred", "action_noise_pred"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                (x, action, timestep, context, state, embodiment_id),
                onnx_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
            )
        print(f"  ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        import traceback
        print(f"  ERROR: ONNX export failed. Exception type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None


def export_to_onnx_14B(pytorch_model, test_inputs, onnx_path="tensorrt/wan_model.onnx"):
    """Export PyTorch model to ONNX"""
    print("Exporting model to ONNX...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pytorch_model.eval()
    pytorch_model.to(torch.float16)

    x, action, timestep, context, state, embodiment_id, clip_feature, y = test_inputs

    # Define input names for better ONNX graph
    input_names = [
        "x",
        "action",
        "timestep",
        "context",
        "state",
        "embodiment_id",
        "clip_feature",
        "y",
    ]
    output_names = ["video_noise_pred", "action_noise_pred"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                (x, action, timestep, context, state, embodiment_id, clip_feature, y),
                onnx_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
            )
        print(f"  ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        import traceback
        print(f"  ERROR: ONNX export failed. Exception type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None


def _parse_trt_shape_spec(shape_spec: str | None) -> dict[str, tuple[int, ...]]:
    if shape_spec is None:
        return {}
    shapes: dict[str, tuple[int, ...]] = {}
    for item in shape_spec.split(","):
        name, raw_shape = item.split(":", 1)
        shapes[name] = tuple(int(dim) for dim in raw_shape.split("x"))
    return shapes


def _set_builder_flag_if_available(config, flag_name: str) -> None:
    if hasattr(trt.BuilderFlag, flag_name):
        try:
            config.set_flag(getattr(trt.BuilderFlag, flag_name))
            print(f"  Enabled TensorRT builder flag: {flag_name}")
        except Exception as exc:
            print(f"  Warning: failed to enable TensorRT builder flag {flag_name}: {exc}")


def _build_tensorrt_engine_python(onnx_path, engine_path, min_shape=None, max_shape=None, opt_shape=None):
    """Build TensorRT engine with the Python API when trtexec is unavailable."""
    print("Building TensorRT engine with Python API fallback...")
    onnx_path = os.path.abspath(onnx_path)
    engine_path = os.path.abspath(engine_path)
    onnx_dir = os.path.dirname(onnx_path)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 0
    if (
        hasattr(trt, "NetworkDefinitionCreationFlag")
        and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH")
    ):
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    old_cwd = os.getcwd()
    try:
        # TensorRT resolves ONNX external-data tensors relative to cwd.
        os.chdir(onnx_dir)
        with open(onnx_path, "rb") as f:
            parsed = parser.parse(f.read())
    finally:
        os.chdir(old_cwd)
    if not parsed:
        print("  ERROR: ONNX parse failed")
        for idx in range(parser.num_errors):
            print(f"    {parser.get_error(idx)}")
        return None

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = 64 << 30

    # Mirror the original trtexec flags (--fp8 --fp16 --bf16) where supported.
    for flag_name in ("FP16", "BF16", "FP8", "FP4"):
        _set_builder_flag_if_available(config, flag_name)

    min_shapes = _parse_trt_shape_spec(min_shape)
    opt_shapes = _parse_trt_shape_spec(opt_shape)
    max_shapes = _parse_trt_shape_spec(max_shape)
    if min_shapes or opt_shapes or max_shapes:
        profile = builder.create_optimization_profile()
        for input_idx in range(network.num_inputs):
            tensor = network.get_input(input_idx)
            name = tensor.name
            shape = tuple(int(dim) for dim in tensor.shape)
            if name in min_shapes:
                profile.set_shape(name, min_shapes[name], opt_shapes[name], max_shapes[name])
                print(
                    f"  Dynamic shape profile for {name}: "
                    f"min={min_shapes[name]} opt={opt_shapes[name]} max={max_shapes[name]}"
                )
            elif any(dim < 0 for dim in shape):
                print(f"  ERROR: dynamic input {name} has no profile shape. Tensor shape: {shape}")
                return None
        config.add_optimization_profile(profile)

    try:
        serialized_engine = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        serialized_engine = engine.serialize() if engine is not None else None

    if serialized_engine is None:
        print("  ERROR: TensorRT Python API failed to build serialized engine")
        return None

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized_engine))
    print(f"  TensorRT engine built successfully: {engine_path}")
    return engine_path


def build_tensorrt_engine(onnx_path, engine_path="tensorrt/wan_model.trt", min_shape=None, max_shape=None, opt_shape=None):
    """Build TensorRT engine from ONNX using trtexec"""
    print("Building TensorRT engine with trtexec...")
    onnx_path = os.path.abspath(onnx_path)
    engine_path = os.path.abspath(engine_path)
    onnx_dir = os.path.dirname(onnx_path)

    if not os.path.exists(onnx_path):
        print(f"  ERROR: ONNX file not found: {onnx_path}")
        return None

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # Build engine using trtexec (much faster than torch_tensorrt)
    trtexec_bin = os.getenv("TRTEXEC_PATH") or shutil.which("trtexec") or "/opt/tensorrt/bin/trtexec"
    if not os.path.exists(trtexec_bin):
        print(
            "  trtexec was not found. Install TensorRT CLI tools or set "
            f"TRTEXEC_PATH to the trtexec binary to use the CLI builder. Tried: {trtexec_bin}"
        )
        return _build_tensorrt_engine_python(onnx_path, engine_path, min_shape, max_shape, opt_shape)
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp8",
        "--fp16",
        "--bf16",
        "--separateProfileRun",
        "--profilingVerbosity=detailed",
        "--memPoolSize=workspace:65536",
        "--dumpProfile",
        "--dumpLayerInfo",
        "--useCudaGraph",
        "--verbose",
    ]

    if min_shape is not None:
        cmd.append(f"--minShapes={min_shape}")
    if max_shape is not None:
        cmd.append(f"--maxShapes={max_shape}")
    if opt_shape is not None:
        cmd.append(f"--optShapes={opt_shape}")

    # Create log file for trtexec output
    log_file = engine_path.replace(".trt", "_build.log")

    try:
        print(f"  Running: {' '.join(cmd)}")
        print(f"  Logging output to: {log_file}")

        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=3600, cwd=onnx_dir)

        if result.returncode == 0:
            print(f"  TensorRT engine built successfully: {engine_path}")
            print(f"  Build log saved to: {log_file}")
            return engine_path
        else:
            print(f"  ERROR: trtexec failed with return code {result.returncode}")
            print(f"  Check build log for details: {log_file}")
            # Print last few lines of log file for immediate feedback
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        print("  Last few lines from build log:")
                        for line in lines[-10:]:  # Show last 10 lines
                            print(f"    {line.rstrip()}")
            except:
                pass
            return None

    except subprocess.TimeoutExpired:
        print("  ERROR: trtexec timed out after 60 minutes")
        print(f"  Partial build log saved to: {log_file}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to run trtexec: {e}")
        return None


def torch_type(trt_type):
    mapping = {
        trt.float32: torch.float32,  # Added missing FLOAT mapping
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(
        f"Could not resolve TensorRT datatype to an equivalent torch datatype. {trt_type}"
    )


class Engine(object):
    def __init__(self, file, plugins=[]):
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        def destroy(self):
            del self.execution_context
            del self.handle

        atexit.register(destroy, self)
        self.print()

    def print(self):

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as f:
            self.handle = runtime.deserialize_cuda_engine(f.read())
            assert (
                self.handle is not None
            ), f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta, self.in_meta, self.out_meta = [], [], []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            print(f"Tensor name: {tensor_name}, shape: {shape}")
            dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])
        self.input_names = {item[0] for item in self.in_meta}

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape):
        if name not in self.input_names:
            return
        self.execution_context.set_input_shape(name, shape)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        reference_tensors = []
        stream = torch.cuda.current_stream()
        for iarg, x in enumerate(args):
            name, shape, dtype = self.in_meta[iarg]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            assert isinstance(x, torch.Tensor), f"Unsupported tensor type: {type(x)}"
            assert runtime_shape == x.shape, f"Invalid input shape: {runtime_shape} != {x.shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor dtype, excepted dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor device, excepted device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, shape, dtype in self.in_meta:
            if name not in kwargs:
                continue

            runtime_shape = self.execution_context.get_tensor_shape(name)
            x = kwargs[name]
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            assert (
                runtime_shape == x.shape
            ), f"Invalid input[{name}] shape: {x.shape}, but the expected shape is: {runtime_shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert (
                x.is_cuda
            ), f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for item in self.out_meta:
            name = item[0]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output_tensor = torch.zeros(
                *runtime_shape, dtype=item[2], device=reference_tensors[0].device
            )
            self.execution_context.set_tensor_address(name, output_tensor.data_ptr())
            reference_tensors.append(output_tensor)

        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert len(reference_tensors) == len(self.in_meta) + len(
            self.out_meta
        ), f"Invalid input tensors. The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, but got {len(reference_tensors)}"

        if return_list:
            return [
                reference_tensors[len(self.in_meta) + i] for i, item in enumerate(self.out_meta)
            ]
        else:
            return {
                item[0]: reference_tensors[len(self.in_meta) + i]
                for i, item in enumerate(self.out_meta)
            }


class WanTrtModel5B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
    ):

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)

        output = self.engine(
            x=x.to(torch.float16),
            action=action.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            state=state.to(torch.float16),
            embodiment_id=embodiment_id.to(torch.int32),
        )
        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()


class WanTrtModel14B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        clip_feature: torch.Tensor,
        y: torch.Tensor,
    ):

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)
        self.engine.set_runtime_tensor_shape("clip_feature", clip_feature.shape)
        self.engine.set_runtime_tensor_shape("y", y.shape)

        output = self.engine(
            x=x.to(torch.float16),
            action=action.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            state=state.to(torch.float16),
            embodiment_id=embodiment_id.to(torch.int32),
            clip_feature=clip_feature.to(torch.float16),
            y=y.to(torch.float16),
        )
        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()


class WanTrtModelAr5B(torch.nn.Module):
    """TRT wrapper for ar_5B_n6 model type - uses kv_cache but no clip_feature."""
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x,
        timestep,
        context,
        kv_cache: list[torch.Tensor],
        y=None,
        action=None,
        timestep_action=None,
        state=None,
    ):

        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("timestep", timestep.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("kv_cache_packed", kv_cache_packed.shape)
        # self.engine.set_runtime_tensor_shape("y", y.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("timestep_action", timestep_action.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)


        output = self.engine(
            x=x.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            kv_cache_packed=kv_cache_packed.to(torch.float16),
            # y.to(torch.float16),
            action=action.to(torch.float16),
            timestep_action=timestep_action.to(torch.float16),
            state=state.to(torch.float16),
        )

        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()


class WanTrtModelAr14B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x,
        timestep,
        context,
        kv_cache: list[torch.Tensor],
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
    ):

        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("timestep", timestep.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("kv_cache_packed", kv_cache_packed.shape)
        self.engine.set_runtime_tensor_shape("y", y.shape)
        self.engine.set_runtime_tensor_shape("clip_feature", clip_feature.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("timestep_action", timestep_action.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)


        output = self.engine(
            x=x.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            kv_cache_packed=kv_cache_packed.to(torch.float16),
            y=y.to(torch.float16),
            clip_feature=clip_feature.to(torch.float16),
            action=action.to(torch.float16),
            timestep_action=timestep_action.to(torch.float16),
            state=state.to(torch.float16),
        )

        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()

def load_tensorrt_engine(engine_path="tensorrt/wan_model.trt", model_type="5B"):
    """Load TensorRT engine"""
    if model_type == "5B":
        trt_inference = WanTrtModel5B(engine_path)
    elif model_type == "ar_5B_n6" or model_type == "ar_5B":
        trt_inference = WanTrtModelAr5B(engine_path)
    elif model_type == "14B":
        trt_inference = WanTrtModel14B(engine_path)
    elif model_type == "ar_14B" or model_type == "ar_14B_droid":
        trt_inference = WanTrtModelAr14B(engine_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    return trt_inference


def create_wan_test_inputs(policy, device="cuda", model_type="5B"):
    # Get dtype from model parameters
    dtype = torch.float16

    # Use hardcoded dimensions from the original working version of the script
    if model_type == "5B":
        x = torch.randn(1, 48, 13, 22, 40, dtype=dtype, device=device)
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        timestep = torch.randn(1, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
        timestep_action = torch.randn(1, 48, dtype=dtype, device=device)
        seq_len = torch.tensor(440, dtype=torch.int32, device=device)
        return x, action, timestep, context, state, embodiment_id, timestep_action, seq_len
    elif model_type == "ar_5B_n6":
        # ar_5B_n6 uses _forward_inference_trt which requires kv_cache_packed
        # Shape from dynamic_axes: kv_cache_packed:30x2x1x220x24x128
        # Note: 5B model doesn't use clip_feature (unlike 14B), but still needs y
        x = torch.randn(1, 48, 2, 22, 40, dtype=dtype, device=device)
        timestep = torch.randn(1, 2, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        # y = torch.randn(1, 52, 2, 22, 40, dtype=dtype, device=device)  # y is required by _forward_inference_trt
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        timestep_action = torch.randn(1, 48, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
    
        num_heads = 24
        head_dim = 128
        num_layers = 30
        B = 1
    
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                torch.zeros([2, B, 13*220, num_heads, head_dim], dtype=dtype, device=device)
            )
    
        kv_cache_packed = torch.stack(kv_cache, dim=0)
        # Return order matches _forward_inference_trt signature: x, timestep, context, kv_cache_packed, y, action, timestep_action, state
        return (x, timestep, context, kv_cache_packed, action, timestep_action, state)
    elif model_type == "14B":
        x = torch.randn(1, 16, 13, 44, 80, dtype=dtype, device=device)
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        timestep = torch.randn(1, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
        clip_feature = torch.randn(1, 257, 1280, dtype=dtype, device=device)
        y = torch.randn(1, 20, 13, 44, 80, dtype=dtype, device=device)
        return x, action, timestep, context, state, embodiment_id, clip_feature, y
    elif model_type == "ar_14B":
        clip_feature = torch.randn(1, 257, 1280, dtype=dtype, device=device)
        y = torch.randn(1, 20, 2, 44, 80, dtype=dtype, device=device)
        timestep_action = torch.randn(1, 48, dtype=dtype, device=device)
        x = torch.randn(1, 16, 2, 44, 80, dtype=dtype, device=device)
        timestep = torch.randn(1, 2, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        seq_len = torch.tensor(1760, dtype=torch.int32, device=device)
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
    
        num_heads = 40 
        head_dim = 5120 // num_heads
        num_layers = 40 
        B = 1
    
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
    
        crossattn_k_cache = []
        for _ in range(num_layers):
            crossattn_k_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
        kv_cache_packed = torch.stack(kv_cache, dim=0)
        crossattn_packed = torch.stack(crossattn_k_cache, dim=0)
        return (x, timestep, context, kv_cache_packed, y, clip_feature, action, timestep_action, state)
    elif model_type == "ar_14B_droid":
        clip_feature = torch.randn(1, 257, 1280, dtype=dtype, device=device)
        y = torch.randn(1, 20, 2, 44, 80, dtype=dtype, device=device)
        timestep_action = torch.randn(1, 24, dtype=dtype, device=device)
        x = torch.randn(1, 16, 2, 44, 80, dtype=dtype, device=device)
        timestep = torch.randn(1, 2, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        seq_len = torch.tensor(1760, dtype=torch.int32, device=device)
        action = torch.randn(1, 24, 32, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
    
        num_heads = 40 
        head_dim = 5120 // num_heads
        num_layers = 40 
        B = 1
    
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
    
        crossattn_k_cache = []
        for _ in range(num_layers):
            crossattn_k_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
        kv_cache_packed = torch.stack(kv_cache, dim=0)
        crossattn_packed = torch.stack(crossattn_k_cache, dim=0)
        return (x, timestep, context, kv_cache_packed, y, clip_feature, action, timestep_action, state)
