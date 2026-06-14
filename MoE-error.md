Error:
 [vaayu_rust] 65 files registered
  [VAAYU-IO] 65 files registered with io_uring
  [Vaayu] Standard safetensors checkpoint → StandardSafetensorsStrategy
  [ERROR] `cache_position` is part of MiniMaxM2Model.forward's signature, but not documented. Make sure to add it to the
  docstring of the function in /home/a/.cache/huggingface/modules/transformers_modules/MiniMaxAI_hyphen__hyphen_MiniMax_hyphen_
  M2_dot_7/4c5ba52a763240c1/modeling_minimax_m2.py.
  [transformers] The `check_model_inputs` decorator is deprecated in favor of `merge_with_config_defaults`.
  [ERROR] `cache_position` is part of MiniMaxM2ForCausalLM.forward's signature, but not documented. Make sure to add it to the
  docstring of the function in /home/a/.cache/huggingface/modules/transformers_modules/MiniMaxAI_hyphen__hyphen_MiniMax_hyphen_
  M2_dot_7/4c5ba52a763240c1/modeling_minimax_m2.py.
  ✅ Attention implementation: flash_attention_2
     Attention module type: MiniMaxM2Attention

  Traceback (most recent call last):
    File "/home/a/code/ai-ml/Vaayu/test_quick.py", line 60, in <module>
      model = AutoModel.from_pretrained(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/a/code/ai-ml/Vaayu/vaayu/auto_model.py", line 104, in from_pretrained
    File "/home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py", line 413, in __init__                                                   
      self.init_model()                                                                                                          
    File "/home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py", line 689, in init_model                                                 
      self.hf_quantizer.preprocess_model(model = self.model, device_map = device_map)                                            
    File "/home/a/code/ai-ml/Vaayu/venv/lib/python3.12/site-packages/transformers/quantizers/base.py", line 171, in              
  preprocess_model                                                                                                               
      self._process_model_before_weight_loading(model, **kwargs)                                                                 
    File "/home/a/code/ai-ml/Vaayu/venv/lib/python3.12/site-packages/transformers/quantizers/quantizer_finegrained_fp8.py", line 
  108, in _process_model_before_weight_loading                                                                                   
      model = replace_with_fp8_linear(                                                                                           
              ^^^^^^^^^^^^^^^^^^^^^^^^                                                                                           
    File "/home/a/code/ai-ml/Vaayu/venv/lib/python3.12/site-packages/transformers/integrations/finegrained_fp8.py", line 803, in 
  replace_with_fp8_linear                                                                                                        
      model.set_submodule(module_name, new_module)                                                                               
    File "/home/a/code/ai-ml/Vaayu/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 809, in set_submodule     
      parent = self.get_submodule(parent_key)                                                                                    
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                                                                                    
    File "/home/a/code/ai-ml/Vaayu/venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 729, in get_submodule     
      raise AttributeError(                                                                                                      
  AttributeError: FP8Experts has no attribute `0`                                                                                
  [vaayu_rust] files unregistered 


Root cause analysis:

  The model config has FP8 quantization. Vaayu calls hf_quantizer.preprocess_model() at line 689 of vaayu_base.py, which triggers
  transformers 5.12's replace_with_fp8_linear. That function iterates all modules and tries to swap nn.Linear layers for FP8
  equivalents via model.set_submodule(path, new_module).

  For a path like model.layers.0.mlp.experts.0.w1, PyTorch's set_submodule calls get_submodule("model.layers.0.mlp.experts.0") to
  find the parent. That navigates by calling getattr(obj, "0") at the numeric segment — which works fine on a standard
  nn.ModuleList but fails on FP8Experts, a custom class the quantizer itself just created to wrap MiniMaxM2Experts.

  Two separate bugs colliding:

  1. replace_with_fp8_linear in transformers 5.12 converts the experts container to a custom FP8Experts class, but FP8Experts
  doesn't implement __getattr__("0") / integer key access
  2. Then the same function tries to navigate INTO FP8Experts to replace its children's Linear layers, which fails because
  getattr(fp8_experts_obj, "0") raises AttributeError

  This is a transformers 5.12 regression in FP8 handling for custom MoE architectures. It didn't exist in 5.5.4 (the version you
  had before).

  Summary: transformers upgraded from 5.5.4 → 5.12.0 to get OutputRecorder, but 5.12's FP8 quantizer is broken for MiniMax-M2's 
  expert structure. The previous error required 5.12; this error is caused BY 5.12.
