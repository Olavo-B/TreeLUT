import re
import os
import json
import shutil
import numpy as np
import time # Needed for header

# --- NEW IMPORTS ---
# Import the new builder, visitors, and parser
from .codeGen import (
    TreeLUTBuilder, VerilogVisitor, VHDLVisitor, HdlModule, 
    Port, Wire, Assignment, ConditionalAssign, ModuleInstance, 
    RawCode, Header, Parameter
)
from .xgbParser import BoostRegressionParser
# --- END NEW IMPORTS ---

class TreeLUTClassifier:
    def __init__(self, xgb_model, w_feature, w_tree, bits_features, pipeline=[0, 0, 0], dir_path="./", style='mux', argmax=False, quantized=False, min=None, max=None):
        
        print("Info: Initializing TreeLUTClassifier...")
        self._folder_created = False
        self._xgb_model = xgb_model
        self._objective = xgb_model.objective
        self._features = xgb_model.get_booster().feature_names
        if(self._features is None):
            self._features = [f"f{i}" for i in range(xgb_model.n_features_in_)]
        self._w_feature = w_feature
        self._w_tree = w_tree
        self._pipeline = pipeline
        self._dir_path = os.path.join(dir_path, 'TreeLUT')
        self._base_score = None
        self._threshold = None
        self._status = 'init'
        self._treelut_model = None # This will hold the *quantized* trees
        self._n_features = xgb_model.n_features_in_
        self._n_trees_per_class = xgb_model.n_estimators
        self._sum_bit_length = None
        self._classes_bias = None
        
        self.parser = None # Will hold the BoostRegressionParser instance

        # 2025-05-22 15:33:30
        # This parameters are new ones create by Olavo Barros
        self._bits_features = bits_features
        self._argmax = argmax
        self._quantized = quantized # Fixed typo: was _quantided
        self._X_min, self._X_max = min, max
        self._style = self._set_style(style)

        if(self._objective == 'binary:logistic'):
            self._n_classes = 1
            config_dict = json.loads(xgb_model.get_booster().save_config())
            base_p = float(config_dict['learner']['learner_model_param']['base_score'])
            self._base_score = np.log(base_p / (1 - base_p))
        elif(self._objective == 'multi:softmax'):
            self._n_classes = xgb_model.n_classes_
        else:
            self._status = 'error'
            print("Error: TreeLUT currently supports the following XGBoost objectives: 'binary:logistic', 'multi:softmax'!")
        
    def convert(self):
        """
        Parses the XGBoost model and quantizes it.
        """
        if(self._status == 'init'):
            # 1. Parse the model using xgbParser
            print("Info: Parsing XGBoost model...")
            self.parser = BoostRegressionParser(self._xgb_model)
            self._treelut_model = self.parser.trees # Get the un-quantized trees
            
            # 2. Quantize the model (modifies self._treelut_model in-place)
            print("Info: Quantizing model...")
            self._quantize_model()
            
            # Note: We don't need to push the trees back into the parser,
            # because the builder will receive the parser *and* the
            # external quantization data (classes_bias, thresholds).
            # Let's re-think: The builder from codeGen.py *does* expect
            # the parser's trees to be the quantized ones.
            # self.parser.trees = self._treelut_model # <-- This is not possible, 'trees' is a property.
            # We must pass the *quantized* self._treelut_model to the builder.
            
            # Let's adjust the builder's __init__ to accept the quantized trees
            # directly, OR adjust our plan.
            
            # --- Simpler Plan ---
            # The builder in `codeGen.py` takes `treelut_model: BaseBoostParser`.
            # Let's just modify the parser's internal model after quantization.
            # `xgbParser.py` shows `self._forest_model`.
            self.parser._forest_model = self._treelut_model
            
            self._status = 'quantized'
            print("Info: Model conversion and quantization complete.")
            
        elif(self._status == 'quantized'):
            print('Info: The model has already been converted!')
        else:
            print('Error!')

    def predict(self, X_test):
        if(self._status == 'quantized'):
            y_pred = self._model_predict(np.array(X_test))
            return y_pred
        else:
            print('Info: Please convert the model into a TreeLUT model first!')

    def verilog(self):
        """
        Generates all Verilog files using TreeLUTBuilder and VerilogVisitor.
        """
        if self._status != 'quantized':
            print('Info: Please convert the model into a TreeLUT model first!')
            return

        print("Info: Starting Verilog generation...")
        if not self._folder_created:
            if os.path.exists(self._dir_path):
                shutil.rmtree(self._dir_path)
            os.makedirs(self._dir_path)
            self._folder_created = True

        verilog_dir = os.path.join(self._dir_path, 'verilog')
        os.makedirs(verilog_dir, exist_ok=True)
        
        # 1. Get thresholds for the builder
        thresholds_list = None
        if self._quantized:
            thresholds_list = self._get_threashold(self._X_min, self._X_max)

        # 2. Instantiate the Builder
        print("Info: Initializing TreeLUTBuilder...")
        builder = TreeLUTBuilder(
            treelut_model=self.parser,
            w_feature=self._w_feature,
            w_tree=self._w_tree,
            bits_features=self._bits_features,
            n_classes=self._n_classes,
            objective=self._objective,
            pipeline=self._pipeline,
            style=self._style,
            argmax=self._argmax,
            quantized=self._quantized,
            min_vals=self._X_min,
            max_vals=self._X_max,
            classes_bias=self._classes_bias,
            thresholds=thresholds_list if self._quantized else [0] * self._n_features
        )

        # 3. Build the AST
        print("Info: Building Hardware AST...")
        modules_ast = builder.build_system()

        # 4. Instantiate the Visitor
        verilog_gen = VerilogVisitor()

        # 5. Generate Verilog code for each module
        print(f"Info: Generating {len(modules_ast)} Verilog modules in {verilog_dir}...")
        for module_name, module_ast in modules_ast.items():
            file_path = os.path.join(verilog_dir, f"{module_name}.v")
            try:
                verilog_code = module_ast.accept(verilog_gen)
                with open(file_path, 'w') as f:
                    f.write(verilog_code)
            except Exception as e:
                print(f"Error generating Verilog for module {module_name}: {e}")
        
        print("Info: Verilog generation complete.")

    def vhdl(self):
        """
        Generates all VHDL files using TreeLUTBuilder and VHDLVisitor.
        """
        if self._status != 'quantized':
            print('Info: Please convert the model into a TreeLUT model first!')
            return

        print("Info: Starting VHDL generation...")
        if not self._folder_created:
            if os.path.exists(self._dir_path):
                shutil.rmtree(self._dir_path)
            os.makedirs(self._dir_path)
            self._folder_created = True

        vhdl_dir = os.path.join(self._dir_path, 'vhdl')
        os.makedirs(vhdl_dir, exist_ok=True)
        
        thresholds_list = None
        if self._quantized:
            thresholds_list = self._get_threashold(self._X_min, self._X_max)

        print("Info: Initializing TreeLUTBuilder...")
        builder = TreeLUTBuilder(
            treelut_model=self.parser,
            w_feature=self._w_feature,
            w_tree=self._w_tree,
            bits_features=self._bits_features,
            n_classes=self._n_classes,
            objective=self._objective,
            pipeline=self._pipeline,
            style=self._style,
            argmax=self._argmax,
            quantized=self._quantized,
            min_vals=self._X_min,
            max_vals=self._X_max,
            classes_bias=self._classes_bias,
            thresholds=thresholds_list if self._quantized else [0] * self._n_features
        )

        print("Info: Building Hardware AST...")
        modules_ast = builder.build_system()

        vhdl_gen = VHDLVisitor()

        print(f"Info: Generating {len(modules_ast)} VHDL modules in {vhdl_dir}...")
        for module_name, module_ast in modules_ast.items():
            file_path = os.path.join(vhdl_dir, f"{module_name}.vhd")
            try:
                vhdl_code = module_ast.accept(vhdl_gen)
                with open(file_path, 'w') as f:
                    f.write(vhdl_code)
            except Exception as e:
                print(f"Error generating VHDL for module {module_name}: {e}")
        
        print("Info: VHDL generation complete.")
            
    def testbench(self, X_test, y_test):
        if(self._folder_created == False):
            if os.path.exists(self._dir_path):
                shutil.rmtree(self._dir_path)
            os.makedirs(self._dir_path)
            self._folder_created = True
        
        if(self._status == 'quantized'):
            os.makedirs(os.path.join(self._dir_path, 'testbench'))
            self._verilog_testbench(np.array(X_test), np.array(y_test))
            self._compile_file()

        else:
            print('Info: Please convert the model into a TreeLUT model first!')
    
    def nodes(self):
        """Returns the number of nodes in each TreeLUT tree.
        """

        if(self._status == 'quantized'):
            nodes_count = [len(tree) for tree in self._treelut_model]
            return nodes_count
        else:
            print('Info: Please convert the model into a TreeLUT model first!')

    @property
    def trees(self):
        """Returns the quantized TreeLUT model as a list of trees.
        """
        if(self._status == 'quantized'):
            return self._treelut_model
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
        
    @property
    def classes_bias(self):
        """Returns the bias of each class in the TreeLUT model.
        """
        if(self._status == 'quantized'):
            return self._classes_bias
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
    
    @property
    def min(self):
        if(self._status == 'quantized'):
            return self._X_min
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
    
    @property
    def max(self):
        if(self._status == 'quantized'):
            return self._X_max
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
    
    @property
    def n_classes(self):
        if(self._status == 'quantized' or self._status == 'init'):
            return self._n_classes
        else:
            print('Info: Please convert the model into a TreeLUT model first!')
    
    @property
    def max_depth(self):
        if(self._status == 'quantized'):
            return self._xgb_model.max_depth
        else:
            print('Info: Please convert the model into a TreeLUT model first!')

    # ===================================================================
    # --- PRIVATE HELPER METHODS (KEPT) ---
    # ===================================================================

    def _set_style(self, style):
        if style not in ['mux', 'equation']:
            raise ValueError("Style must be either 'mux' or 'equation'.")
        return style
        
    def _bitwidth(self, N):
        if N >= 0:
            return int(np.floor(np.log2(N)) + 1) if N > 0 else 1
        else: # Handle negative numbers for bias/threshold
            return int(np.floor(np.log2(abs(N))) + 2) if N != -1 else 1

    # --- MODIFIED: To iterate dict values ---
    def _minmax_trees(self):
        min_trees = np.zeros((len(self._treelut_model),))
        max_range = -np.inf
        for i, tree in enumerate(self._treelut_model): # tree is dict[str, dict]
            all_values = np.array([d['value'] for d in tree.values() if 'value' in d])
            if len(all_values) == 0:
                min_trees[i] = 0
                continue
            min_trees[i] = all_values.min()
            max_range = max((all_values.max()-all_values.min()), max_range)
        return([min_trees, max_range])

    # --- MODIFIED: To iterate dict values and work with string keys ---
    def _quantize_model(self):
        # self._treelut_model is list[dict[str, dict]] from parser
        trees_bias = np.zeros((self._n_classes, self._n_trees_per_class))
        [min_trees, max_range] = self._minmax_trees()
        
        if max_range == 0:
            scale = 1.0 # Avoid division by zero
        else:
            scale = (2 ** self._w_tree - 1) / (max_range)
            
        for i in range(len(self._treelut_model)): # Iterate over trees
            trees_bias[i%self._n_classes][int(np.floor(i/self._n_classes))] = min_trees[i]
            
            # Iterate over nodes in the tree dict
            for node_id_str, node in self._treelut_model[i].items(): 
                if(node['type'] == 'leaf'):
                    node['value'] = int(np.round((node['value'] - min_trees[i])*scale))
                else:
                    # Quantize threshold
                    node['threshold'] = int(np.ceil(node['threshold']))
                    
                    # Add parent links (needed by builder 'equation' style)
                    yes_node_str = str(node['yes'])
                    no_node_str = str(node['no'])
                    
                    if yes_node_str in self._treelut_model[i]:
                        self._treelut_model[i][yes_node_str]['parent_node'] = node_id_str
                        self._treelut_model[i][yes_node_str]['parent_yesno'] = 'yes'
                    if no_node_str in self._treelut_model[i]:
                        self._treelut_model[i][no_node_str]['parent_node'] = node_id_str
                        self._treelut_model[i][no_node_str]['parent_yesno'] = 'no'
        
        if(self._n_classes == 1):
            self._threshold = int(np.round(((-np.sum(min_trees))-self._base_score)*scale))
            
        self._classes_bias = trees_bias.sum(axis=1)
        self._classes_bias = self._classes_bias - self._classes_bias.min()
        self._classes_bias = (np.round((self._classes_bias) * scale)).astype(int)

    def _single_tree_predict(self, tree, X):
        # This function expects a numpy array, let's adapt it to dict
        # Or... let's check _model_predict
        
        # --- This logic is complex, let's adapt _model_predict instead ---
        
        # The original _single_tree_predict assumes a numpy array structure
        # that was created *inside* _model_predict. We should keep
        # _model_predict and _single_tree_predict as they were.
        
        node_id = np.zeros((X.shape[0], )).astype(int)
        
        # We need the numpy-formatted tree here.
        # Let's re-create it just for this function.
        tree_numpy = self._convert_tree_to_numpy(tree)
        
        is_leaf = (tree_numpy[node_id, 1] == 1).astype(bool)
        while np.any(is_leaf == False):
            feature = tree_numpy[node_id, 2].astype(int)
            threshold = tree_numpy[node_id, 3]
            comparison = X[np.arange(X.shape[0]), feature] < (threshold-(1e-8))
            node_id[(~is_leaf) & (comparison == True)] = tree_numpy[node_id[(~is_leaf) & (comparison == True)], 4]
            node_id[(~is_leaf) & (comparison == False)] = tree_numpy[node_id[(~is_leaf) & (comparison == False)], 5]
            is_leaf = (tree_numpy[node_id, 1] == 1)
        return tree_numpy[node_id, 3]
    
    def _convert_tree_to_numpy(self, tree_dict):
        """Converts a single tree_dict to the numpy format _model_predict expects."""
        
        # Find max node ID to size array
        max_id = 0
        for node_id_str in tree_dict.keys():
            max_id = max(max_id, int(node_id_str))
            
        tree_numpy = np.zeros((max_id + 1, 6)).astype(int)
        
        for node_id_str, node in tree_dict.items():
            j = int(node_id_str)
            if(node['type'] == 'leaf'):
                tree_numpy[j, 0] = j # node id
                tree_numpy[j, 1] = 1 # node type (leaf)
                tree_numpy[j, 3] = node['value'] # leaf value
            else:
                tree_numpy[j, 0] = j # node id
                tree_numpy[j, 1] = 0 # node type (split)
                tree_numpy[j, 2] = node['feature'] # feature
                tree_numpy[j, 3] = node['threshold'] # threshold
                tree_numpy[j, 4] = int(node['yes']) # yes val
                tree_numpy[j, 5] = int(node['no']) # no val
        return tree_numpy

    def _model_predict(self, X_test):
        predictions = np.zeros((X_test.shape[0], self._n_classes))
        for i in range(self._n_classes):
            predictions[:, i] = self._classes_bias[i]
        
        for i, tree in enumerate(self._treelut_model): # self._treelut_model is list[dict]
            class_number = i%self._n_classes
            # We must use the dict-based tree
            predictions[:, class_number] += self._single_tree_predict_dict(tree, X_test)
                
        if(self._n_classes == 1):
            return predictions >= (self._threshold)
        else:
            return np.argmax(predictions, axis=1)

    def _single_tree_predict_dict(self, tree, X):
        """Software prediction using the dict-based tree structure."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            node_id = '0' # Start at root
            while tree[node_id]['type'] != 'leaf':
                node = tree[node_id]
                feature_idx = node['feature']
                if X[i, feature_idx] < node['threshold']:
                    node_id = str(node['yes'])
                else:
                    node_id = str(node['no'])
            predictions[i] = tree[node_id]['value']
        
        return predictions

    def _get_threashold(self, X_min, X_max) -> np.ndarray:
        """
        Calculate the threshold for the quantization module based on the minimum and maximum values of the features.
        """
        if X_min is None or X_max is None:
            print("Warning: Quantization min/max values not provided. Using dummy thresholds.")
            return np.zeros((self._n_features, 2**self._w_feature - 1))

        thresholds = np.zeros((self._n_features, 2**self._w_feature - 1))
        for feature_idx in range(self._n_features):
            min_val = X_min[feature_idx]
            max_val = X_max[feature_idx]
            if min_val == max_val - 1:
                thresholds[feature_idx, :] = -1
                continue
            if max_val - min_val <= 2**self._w_feature - 1:
                thresholds[feature_idx, 0] = min_val+1
                for j in range(1,  max_val - min_val):
                    thresholds[feature_idx, j] = thresholds[feature_idx, j-1] + 1
                thresholds[feature_idx, max_val - min_val:] = -1
            else:
                step = int((max_val - min_val) / (2**self._w_feature - 1))
                thresholds[feature_idx, 0] = int(min_val + step / 2)
                for j in range(1, 2**self._w_feature - 1):
                    thresholds[feature_idx, j] = thresholds[feature_idx, j-1] + step
        return thresholds

    def _int_2_bitstring(self, value, bits):
        if value < 0:
            value = (1 << bits) + value
        result = format(value, f'0{bits}b')
        return f"{bits}'b{result}"

    # --- TESTBENCH METHODS (KEPT AS-IS) ---
        
    def _verilog_testbench(self, X_test, y_test):
        with open(os.path.join(self._dir_path, 'testbench/X_test.mem'), 'w') as f:
            for key in X_test:
                for value in np.flip(key):
                    x_val = value
                    if self._quantized:
                        binary_string = format(int(x_val), f'0{self._bits_features}b')
                    else:
                        binary_string = format(int(x_val), f'0{self._w_feature}b')
                    f.write(binary_string)
                f.write("\n")
                
        with open(os.path.join(self._dir_path, 'testbench/y_test.mem'), 'w') as f:
            n_output_bits = self._bitwidth(self._n_classes-1)
            for value in y_test:
                binary_string = format(int(value), f'0{n_output_bits}b')
                f.write(binary_string)
                f.write("\n")
                
        with open(os.path.join(self._dir_path, 'testbench/testbench.v'), 'w') as f:
            n_output_bits = self._bitwidth(self._n_classes-1)
            
            # Use self._quantized (fixed typo)
            if self._quantized:
                f.write(f"`timescale 1ns/1ps\n\nmodule sim ();\n\nreg [{self._n_features*self._bits_features-1}:0] x_test [0:{len(y_test)-1}];\n")
                tb_input_wire = f"reg [{self._n_features*self._bits_features-1}:0] treelut_input;\n"
                tb_input_port = "all_features"
            else:
                f.write(f"`timescale 1ns/1ps\n\nmodule sim ();\n\nreg [{self._n_features*self._w_feature-1}:0] x_test [0:{len(y_test)-1}];\n")
                tb_input_wire = f"reg [{self._n_features*self._w_feature-1}:0] treelut_input;\n"
                tb_input_port = "i"
                
            f.write(f"reg [{n_output_bits-1}:0] y_test [0:{len(y_test)-1}];\n\n")
            
            f.write(tb_input_wire)
            
            f.write(f"wire [{n_output_bits-1}:0] out_predicted;\n\n")            
            f.write(f"reg [{n_output_bits-1}:0] out_expected;\n\n")
            
            f.write(f"integer i, j;\n")
            
            n_address_bits = self._bitwidth(len(y_test)-1)
            f.write(f"reg [{n_address_bits-1}:0] result_false;\n")
            f.write(f"reg [{n_address_bits-1}:0] result_true;\n")
            
            if(self._pipeline[0] == 0 and self._pipeline[1] == 0 and self._pipeline[2] == 0):
                f.write(f"TreeLUT TreeLUT_inst(.{tb_input_port}(treelut_input), .o(out_predicted));\n")
            else:
                f.write(f"reg clk;\n\n")
                f.write(f"TreeLUT TreeLUT_inst(.clk(clk), .{tb_input_port}(treelut_input), .o(out_predicted));\n")

            f.write(f"initial begin\n\t\n\tresult_false <= 0; result_true <= 0;\n\t$readmemb(\"X_test.mem\", x_test); $readmemb(\"y_test.mem\", y_test);\nend\n\n")
            
            if(self._pipeline[0] != 0 or self._pipeline[1] != 0 or self._pipeline[2] != 0):
                f.write(f"initial begin\n\tclk <= 0;\n\tforever #5 clk <= ~clk;\nend\n\n")
    
            f.write(f"initial begin\n\t#105\n\tfor (i = 0; i <= {len(y_test)}; i = i + 1) begin\n\t\t#10\n\t\ttreelut_input <= x_test[i];\n\tend\nend\n\n")
            
            f.write(f"initial begin\n\t#{105+10*(self._pipeline[0]+self._pipeline[1]+self._pipeline[2])}\n")
            f.write(f"\tfor (j = 0; j <= {len(y_test)}; j = j + 1) begin\n\t\t#10\n\t\tout_expected <= y_test[j];\n")
            f.write(f"\t\tif(out_predicted == out_expected)\n\t\t\tresult_true <= result_true + 1;\n\t\telse\n\t\t\tresult_false <= result_false+1;\n\tend\n")
            f.write(f"\t#10\n\t$display(\"Result: %d/%d\", result_true, result_false);\n")
            f.write(f"\t#10\n\t$display(\"Accuracy: %f%%\", (result_true*100.0)/(result_true+result_false));\n")
            f.write(f"\t$finish;\nend\nendmodule")
            
            f.close()

    def _compile_file(self):
        """
        .txt file with the path to all verilog files to be compiled.
        """
        with open(os.path.join(self._dir_path, 'testbench/compile.txt'), 'w') as f:
            f.write(f"{self._dir_path}/testbench/testbench.v\n")
            for root, dirs, files in os.walk(os.path.join(self._dir_path, 'verilog')):
                for file in files:
                    if file.endswith('.v'):
                        f.write(os.path.join(root, file) + "\n")
            f.write("\n")

    # ===================================================================
    # --- DELETED METHODS ---
    # _format_model (replaced by xgbParser)
    # _node_regex (replaced by xgbParser)
    # _leaf_regex (replaced by xgbParser)
    # _extract_unique_features (logic moved to TreeLUTBuilder)
    # _verilog_unique_features (replaced by Visitor)
    # _verilog_trees (replaced by Builder+Visitor)
    # _verilog_addertree (replaced by Builder+Visitor)
    # _verilog_adder (replaced by Builder+Visitor)
    # _verilog_ports (replaced by Builder+Visitor)
    # _verilog_myreg (replaced by Builder+Visitor)
    # _argmax_module (replaced by Builder+Visitor)
    # _quantization_module (replaced by Builder+Visitor)
    # ===================================================================