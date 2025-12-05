# /*****************************************************************************/
#  * File: codeGen.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-10-27
#  * License: [License Type]
#  * Description: Class for code generation (Verilog and VHDL)
# /*****************************************************************************/


import re
import time
import numpy as np

from .xgbParser import BaseBoostParser

#===============================
# HDL AST Node Definitions
#===============================

"""
Defines the language-agnostic Abstract Syntax Tree (AST) nodes
for our hardware design.
"""

class HdlNode:
    """Base class for all AST nodes."""
    def accept(self, visitor):
        # Dynamically call the correct visit method on the visitor
        method_name = f'visit_{self.__class__.__name__.lower()}'
        if hasattr(visitor, method_name):
            return getattr(visitor, method_name)(self)
        else:
            raise NotImplementedError(f"Visitor {visitor.__class__.__name__} does not have method {method_name}")

class HdlModule(HdlNode):
    def __init__(self, name):
        self.name = name
        self.ports = []
        self.wires = []
        self.assignments = []
        self.instances = []
        self.headers = []
        self.raw_code_blocks = []
        self.parameters = []

    def add_port(self, port): self.ports.append(port)
    def add_wire(self, wire): self.wires.append(wire)
    def add_assignment(self, assign): self.assignments.append(assign)
    def add_instance(self, inst): self.instances.append(inst)
    def add_raw_code(self, raw_code): self.raw_code_blocks.append(raw_code)
    def add_parameter(self, param): self.parameters.append(param)
    def add_header(self, header): self.headers.append(header)

class ComparatorModule(HdlModule):
    def __init__(self, name):
        super().__init__(name)
        
            


class Port(HdlNode):
    def __init__(self, name, direction, width, is_reg=False):
        self.name = name
        self.direction = f'{"input wire" if direction == "in" else "output wire"}'
        self.width = width
        self.is_reg = is_reg

class Wire(HdlNode):
    def __init__(self, name, width, is_reg=False):
        self.name = name
        self.width = width
        self.is_reg = is_reg

class Assignment(HdlNode):
    """Represents a simple assign: target = expression"""
    def __init__(self, target, expression):
        self.target = target         # e.g., "my_wire" (a string)
        self.expression = expression # e.g., "other_wire | another" (a string)

class ConditionalAssign(Assignment):
    """Represents a Mux: target = cond ? true_val : false_val"""
    def __init__(self, target, condition, true_val, false_val):
        self.target = target
        self.condition = condition
        self.true_val = true_val
        self.false_val = false_val

class ArithmeticAssign(Assignment):
    """Represents an addition: target = op1 + op2"""
    def __init__(self, target, operand1, operand2, ta_width=None, op_width1=None, op_width2=None, operation='+'):
        self.target = target
        self.operand1 = operand1
        self.operand2 = operand2

        # NOTE - 2025-12-03 15:03:26: These are due the difference in bit-width
        # handling VHDL needs the same width for both operands and target
        self.ta_width = ta_width
        self.op_width1 = op_width1
        self.op_width2 = op_width2
        self.operation = operation


class ModuleInstance(HdlNode):
    def __init__(self, name, module_type, port_map, param_map=None):
        self.name = name               # "tree_0_inst"
        self.module_type = module_type # "tree_0"
        self.port_map = port_map     # {"i": "comparators_wire", "o": "tree_0_out"}
        self.param_map = param_map   # {"DataWidth": 16}

class ComparatorInstance(ModuleInstance):
    """Represents a comparison: target = out1 if (comparator < threshold) else out2
    or a call to an external comparator module."""
    def __init__(self, name, port_map, target, comparator, threshold, param_map=None):
        super().__init__(name=name, 
                         module_type="comparator_operator", 
                         port_map=port_map, 
                         param_map=param_map)
        self.target = target
        self.comparator = comparator
        self.threshold = threshold

class RawCode(HdlNode):
    def __init__(self, verilog_code, vhdl_code):
        self.verilog_code = verilog_code
        self.vhdl_code = vhdl_code

class Header(HdlNode):
    def __init__(self, comment):
        self.comment = comment

class Parameter(HdlNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value

#===============================
# TreeLUT Code Generation Class
#===============================

class TreeLUTBuilder:
    """
    Builds a language-agnostic Hardware Abstract Syntax Tree (AST)
    from a parsed and quantized TreeLUT model.

    This class is responsible for generating the abstract hardware
    structure (modules, wires, adders, etc.) but *not* for
    generating Verilog/VHDL syntax.
    """

    def __init__(self, treelut_model: BaseBoostParser,
                 w_feature: int, w_tree: int, bits_features: int,
                 n_classes: int,
                 objective: str,
                 pipeline: list = [0, 0, 0],
                 style: str = 'mux',
                 argmax: bool = False,
                 quantized: bool = False,
                 min_vals: list = None,
                 max_vals: list = None,
                 classes_bias: list = None,
                 thresholds: list = [0]):

        # Parameters for code generation
        self._w_feature = w_feature
        self._w_tree = w_tree
        self._bits_features = bits_features
        self._pipeline = pipeline
        self._style = self._set_style(style)
        self._argmax = argmax
        self._quantized = quantized
        self._X_min, self._X_max = min_vals, max_vals

        # Model parameters
        if n_classes is None or n_classes == 0:
            if "multi" in objective:
                 raise ValueError("n_classes must be provided for multiclass objective")
            self._n_classes = 1
        else:
            self._n_classes = n_classes

        self._treelut_model_struct = treelut_model.trees
        self._n_features = treelut_model.n_features
        self._n_trees_per_class = len(self._treelut_model_struct) // self._n_classes

        # Pre-calculated quantization results
        self._classes_bias = classes_bias if classes_bias is not None else [0] * self._n_classes
        self._thresholds = thresholds

        # Internal data
        self._sum_bit_length = None
        self._unique_features = []
        self._trees_bit_length = None # np.array

        # Pre-process model to add parent links (for 'equation' style)
        self._treelut_model_struct = self._link_parents(self._treelut_model_struct)

    # ==============================================================
    # Public Build Method
    # ==============================================================

    def build_system(self):
        """
        Orchestrates the construction of the entire hardware system.
        Returns a dictionary of all HdlModule (modules) to be generated.
        """

        self._unique_features = self._extract_unique_features()

        modules = {}
        top_module = HdlModule("TreeLUT")

        if self._quantized:
            q_module = self._build_quantization_module()
            modules[q_module.name] = q_module
            self._build_quantization_instantiation(top_module)

        tree_modules = self._build_tree_modules()
        for mod in tree_modules:
            modules[mod.name] = mod

        self._calculate_sum_bit_length()

        adder_modules = self._build_adder_modules()
        for mod in adder_modules:
            modules[mod.name] = mod

        if any(p > 0 for p in self._pipeline):
            reg_module = self._build_myreg_module()
            modules[reg_module.name] = reg_module
        
        comparator_modules = self._build_comparator_modules()
        comparator_operator = self._build_one_comparator_module()
        modules[comparator_operator.name] = comparator_operator
        modules[comparator_modules.name] = comparator_modules

        if self._argmax:
            argmax_module = self._build_argmax_module()
            modules[argmax_module.name] = argmax_module

        self._build_top_level_logic(top_module)
        modules[top_module.name] = top_module

        return modules

    # ==============================================================
    # Top-Level Logic Construction
    # ==============================================================

    def _build_top_level_logic(self, top_module):
        """Populates the 'TreeLUT' module with ports, wires, and instances."""

        top_module.add_header(Header(f"""
/*****************************************************************************\\
//                  ___           ___           ___                         ___                 
//      ___        /  /\         /  /\         /  /\                       /__/\          ___   
//     /  /\      /  /::\       /  /:/_       /  /:/_                      \  \:\        /  /\  
//    /  /:/     /  /:/\:\     /  /:/ /\     /  /:/ /\    ___     ___       \  \:\      /  /:/  
//   /  /:/     /  /:/~/:/    /  /:/ /:/_   /  /:/ /:/_  /__/\   /  /\  ___  \  \:\    /  /:/   
//  /  /::\    /__/:/ /:/___ /__/:/ /:/ /\ /__/:/ /:/ /\ \  \:\ /  /:/ /__/\  \__\:\  /  /::\   
// /__/:/\:\   \  \:\/:::::/ \  \:\/:/ /:/ \  \:\/:/ /:/  \  \:\  /:/  \  \:\ /  /:/ /__/:/\:\  
// \__\/  \:\   \  \::/~~~~   \  \::/ /:/   \  \::/ /:/    \  \:\/:/    \  \:\  /:/  \__\/  \:\ 
//      \  \:\   \  \:\        \  \:\/:/     \  \:\/:/      \  \::/      \  \:\/:/        \  \:\\
//       \__\/    \  \:\        \  \::/       \  \::/        \__\/        \  \::/          \__\/
//                 \__\/         \__\/         \__\/                       \__\/                
//        
// Implementation generated by TreeLUT - {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}

INFO:

Model Summary:
- Number of features: {self._n_features}
- Number of classes: {self._n_classes}
- Trees per class: {self._n_trees_per_class}
- Sample bit-width: {self._bits_features}
- Unique comparisons: {len(self._unique_features)}

Hardware Configuration:
- Feature weight: {self._w_feature}
- Tree output bit-width: {self._w_tree}
- Sum output bit-width: {self._sum_bit_length}
- Pipeline stages: {self._pipeline}
- Style: {self._style}
- Quantized input: {self._quantized}
- Argmax output: {self._argmax}

This implementation is provided under the terms of the [License Type] license.
This tool is a product of Olavo Barros(olavo.barros@ufv.br) as a independent
version of the original TreeLUT project (https://doi.org/10.48550/arXiv.2501.01511)

*****************************************************************************/
""".lstrip('\n')))

        n_output_bits = self._bitwidth(self._n_classes - 1)

        if any(p > 0 for p in self._pipeline):
            top_module.add_port(Port("clk", "in", 1))

        if self._quantized:
            in_width = self._n_features * self._bits_features
            top_module.add_port(Port("all_features", "in", in_width))
            top_module.add_wire(Wire("i", self._n_features * self._w_feature))
        else:
            in_width = self._n_features * self._w_feature
            top_module.add_port(Port("i", "in", in_width))

        if self._argmax:
            out_width = n_output_bits
            top_module.add_port(Port("o", "out", out_width))
            top_module.add_wire(Wire("treelut_output", self._n_classes * self._sum_bit_length))
        else:
            out_width = self._n_classes * self._sum_bit_length
            top_module.add_port(Port("o", "out", out_width))

        self._build_comparators_instantiations(top_module)
        self._build_tree_instantiations(top_module)
        self._build_adder_instantiations(top_module)

        if self._argmax:
            top_module.add_instance(ModuleInstance(
                name="argmax_inst",
                module_type="argmax",
                port_map={"i": "treelut_output", "o": "o"}
            ))

    def _build_comparator_logic(self, top_module):
        n_comparators = len(self._unique_features)
        top_module.add_wire(Wire("binary_features", n_comparators))

        feature_source = "i"

        for i, (feature_index, threshold) in enumerate(self._unique_features):
            feature_input_bits = f"{(feature_index + 1) * self._w_feature - 1}:{feature_index * self._w_feature}"

            # TODO - 2025-12-02 17:57:09: Change this expression to handle VHDL and Verilog
            expr = (f"({feature_source}[{feature_input_bits}] < "
                    f"({self._w_feature}'d{int(threshold)})) ? 1'b1 : 1'b0")

            top_module.add_assignment(Assignment(
                target=f"binary_features[{i}]",
                expression=expr
            ))

        if self._pipeline[0] != 0:
            top_module.add_wire(Wire("binary_features_reg", n_comparators))
            top_module.add_instance(ModuleInstance(
                name="feature_reg",
                module_type="myreg",
                param_map={"DataWidth": n_comparators},
                port_map={
                    "clk": "clk",
                    "data_in": "binary_features",
                    "data_out": "binary_features_reg"
                }
            ))

    def _build_tree_instantiations(self, top_module):
        total_tree_bits = self._trees_bit_length.sum().sum()
        top_module.add_wire(Wire("trees_output", total_tree_bits))

        comparator_source = "binary_features_reg" if self._pipeline[0] != 0 else "binary_features"

        idx = 0
        for i in range(self._n_classes):
            for j in range(self._n_trees_per_class):
                tree_name = f"class{i}_tree{j}"
                bits = self._trees_bit_length[i, j]

                top_module.add_instance(ModuleInstance(
                    name=f"{tree_name}_inst",
                    module_type=tree_name,
                    port_map={
                        "i": comparator_source,
                        "o": f"trees_output[{idx + bits - 1}:{idx}]"
                    }
                ))
                idx += bits

        if self._pipeline[1] != 0:
            top_module.add_wire(Wire("trees_output_reg", total_tree_bits))
            top_module.add_instance(ModuleInstance(
                name="trees_reg",
                module_type="myreg",
                param_map={"DataWidth": total_tree_bits},
                port_map={
                    "clk": "clk",
                    "data_in": "trees_output",
                    "data_out": "trees_output_reg"
                }
            ))

    def _build_adder_instantiations(self, top_module):
        tree_output_source = "trees_output_reg" if self._pipeline[1] != 0 else "trees_output"
        adder_clk_port = {"clk": "clk"} if self._pipeline[2] != 0 else {}

        adder_output_target = "treelut_output" if self._argmax else "o"

        flat_bit_length = self._trees_bit_length.reshape(-1, order='C')

        for i in range(self._n_classes):
            start_idx = (i * self._n_trees_per_class)
            end_idx = (i + 1) * self._n_trees_per_class

            trees_range = (f"{flat_bit_length[0:end_idx].sum() - 1}:"
                           f"{flat_bit_length[0:start_idx].sum()}")

            port_map = {
                "i": f"{tree_output_source}[{trees_range}]",
                "o": (f"{adder_output_target}[{(i + 1) * self._sum_bit_length - 1}:"
                      f"{i * self._sum_bit_length}]")
            }
            port_map.update(adder_clk_port)

            top_module.add_instance(ModuleInstance(
                name=f"class{i}_adder_inst",
                module_type=f"class{i}_adder",
                port_map=port_map
            ))

    def _build_quantization_instantiation(self, top_module):
        top_module.add_instance(ModuleInstance(
            name="quantization_inst",
            module_type="quantization",
            port_map={
                "i": "all_features",
                "o": "i"
            }
        ))

    def _build_comparators_instantiations(self, top_module):
        top_module.add_wire(Wire("binary_features", len(self._unique_features)))
        top_module.add_instance(ModuleInstance(
            name="comparators_inst",
            module_type="comparator",
            port_map={
                "i": "i",
                "o": "binary_features"
            }
        ))
    # ==============================================================
    # Sub-Module Construction Methods
    # ==============================================================

    def _build_tree_modules(self):
        tree_modules = []
        self._trees_bit_length = np.zeros((self._n_classes, self._n_trees_per_class), dtype=int)

        for tree_idx, tree in enumerate(self._treelut_model_struct):
            if self._style == 'mux':
                mod, path_max = self._build_one_tree_module_mux(tree_idx, tree)
            elif self._style == 'equation':
                mod, path_max = self._build_one_tree_module_equation(tree_idx, tree)

            path_output_bit = self._bitwidth(path_max)
            class_num = tree_idx % self._n_classes
            tree_num = int(np.floor(tree_idx / self._n_classes))
            self._trees_bit_length[class_num, tree_num] = path_output_bit

            mod.ports[1].width = path_output_bit
            if self._style == 'mux':
                #NOTE - 2025-12-05 08:33:57: Mux propagates output not the 
                # expressions result
                for w in mod.wires:
                    w.width = path_output_bit


            tree_modules.append(mod)

        return tree_modules

    def _build_one_tree_module_mux(self, tree_idx, tree):
        class_num = tree_idx % self._n_classes
        tree_num = int(np.floor(tree_idx / self._n_classes))
        tree_name = f"class{class_num}_tree{tree_num}"

        module = HdlModule(tree_name)
        module.add_port(Port("i", "in", len(self._unique_features)))
        module.add_port(Port("o", "out", 1)) # Width updated later

        path_max = 0
        root_node_id = '0'
        queue = [(root_node_id, '')]
        visited_nodes = set()

        while queue:
            current_idx_str, _ = queue.pop(0)
            if current_idx_str in visited_nodes:
                continue

            current_node = tree[current_idx_str]
            if current_node['type'] == 'leaf':
                continue

            visited_nodes.add(current_idx_str)

            yes_node_id_str = str(current_node['yes'])
            no_node_id_str = str(current_node['no'])
            yes_node = tree[yes_node_id_str]
            no_node = tree[no_node_id_str]

            parent_feature = (current_node['feature'], current_node['threshold'])
            unique_feature_idx = self._unique_features.index(parent_feature)
            comparator = f"i[{unique_feature_idx}]"

            if yes_node['type'] == 'leaf':
                yes_expr = f"{self._int_2_bitstring(int(yes_node['value']), self._w_tree)}"
                path_max = max(path_max, int(yes_node['value']))
            else:
                yes_expr = f"new_{yes_node_id_str}"
                queue.append((yes_node_id_str, ''))

            if no_node['type'] == 'leaf':
                no_expr = f"{self._int_2_bitstring(int(no_node['value']), self._w_tree)}"
                path_max = max(path_max, int(no_node['value']))
            else:
                no_expr = f"new_{no_node_id_str}"
                queue.append((no_node_id_str, ''))

            if current_idx_str == root_node_id:
                target = "o"
            else:
                target = f"new_{current_idx_str}"
                module.add_wire(Wire(target, 1)) # Width updated later

            module.add_assignment(ConditionalAssign(
                target=target,
                condition=comparator,
                true_val=yes_expr,
                false_val=no_expr
            ))

        return module, path_max

    def _build_one_tree_module_equation(self, tree_idx, tree):
        class_num = tree_idx % self._n_classes
        tree_num = int(np.floor(tree_idx / self._n_classes))
        tree_name = f"class{class_num}_tree{tree_num}"

        module = HdlModule(tree_name)
        module.add_port(Port("i", "in", len(self._unique_features)))
        module.add_port(Port("o", "out", 1)) # Width updated later

        path_encoded = [[] for _ in range(2**self._w_tree)]
        all_leaves = [(idx, node['value']) for idx, node in tree.items() if node['type'] == 'leaf']

        path_max = 0
        for leaf_idx_str, leaf_value in all_leaves:
            leaf_value_int = int(leaf_value)
            path_max = max(path_max, leaf_value_int)
            node_idx_str = leaf_idx_str
            path_parts = []

            while 'parent_node' in tree[node_idx_str]: # Check avoids root
                parent_idx_str = tree[node_idx_str]['parent_node']
                parent_yesno = tree[node_idx_str]['parent_yesno']
                parent_node = tree[parent_idx_str]

                parent_feature = (parent_node['feature'], parent_node['threshold'])
                unique_feature_idx = self._unique_features.index(parent_feature)

                if parent_yesno == 'yes':
                    path_parts.append(f"i[{unique_feature_idx}]")
                else:
                    path_parts.append(f"(~i[{unique_feature_idx}])")
                node_idx_str = parent_idx_str

            path_str = ' & '.join(reversed(path_parts)) if path_parts else "1'b1"
            path_encoded[leaf_value_int].append(path_str)

        #NOTE - 2025-12-03 16:16:21: Change to get binary values in the final mux of each tree equation
        final_expr = f"{self._int_2_bitstring(path_max, self._bitwidth(path_max))}"
        for score_idx in range(path_max):
             paths = path_encoded[score_idx]
             if not paths:
                 continue

             wire_name_score = f"new_{score_idx}"
             module.add_wire(Wire(wire_name_score, 1))

             path_wires = []
             for path_idx, path_str in enumerate(paths):
                 wire_name_path = f"new_{score_idx}_{path_idx}"
                 module.add_wire(Wire(wire_name_path, 1))
                 module.add_assignment(Assignment(wire_name_path, path_str))
                 path_wires.append(wire_name_path)

             or_expression = " | ".join(path_wires)
             module.add_assignment(Assignment(wire_name_score, or_expression))

             #NOTE - 2025-12-03 16:16:21: Change to get binary values in the final mux of each tree equation
             final_expr = f"({wire_name_score} ? {self._int_2_bitstring(score_idx, self._bitwidth(path_max))} : {final_expr})"

        module.add_assignment(Assignment("o", final_expr))

        return module, path_max

    def _build_adder_modules(self):
        adder_modules = []
        flat_bit_length = self._trees_bit_length.reshape(-1, order='C')

        for i in range(self._n_classes):
            module = HdlModule(f"class{i}_adder")
            class_bias = self._classes_bias[i]

            trees_bits_for_class = flat_bit_length[i * self._n_trees_per_class : (i + 1) * self._n_trees_per_class]

            if self._pipeline[2] != 0:
                module.add_port(Port("clk", "in", 1))
            module.add_port(Port("i", "in", trees_bits_for_class.sum()))
            module.add_port(Port("o", "out", self._sum_bit_length))

            operands_name = []
            operands_bits = []

            if class_bias != 0:
                n_bias_bits = self._bitwidth(class_bias)
                operands_name.append(f"{n_bias_bits}'d{class_bias}")
                operands_bits.append(n_bias_bits)

            idx = 0
            for j in range(self._n_trees_per_class):
                bits = trees_bits_for_class[j]
                operands_name.append(f"i[{idx + bits - 1}:{idx}]")
                operands_bits.append(bits)
                idx += bits

            n_stage = 0

            while len(operands_name) > 1:
                new_operands_name = []
                new_operands_bits = []

                for k in range(int(len(operands_name) / 2)):
                    op1_name, op1_bits = operands_name[2 * k], operands_bits[2 * k]
                    op2_name, op2_bits = operands_name[2 * k + 1], operands_bits[2 * k + 1]

                    n_bits_addition = max(op1_bits, op2_bits) + 1
                    wire_name = f"stage{n_stage}_adder{k}"

                    module.add_wire(Wire(wire_name, n_bits_addition))
                    module.add_assignment(ArithmeticAssign(wire_name, 
                                                      ta_width=n_bits_addition,
                                                        operand1=op1_name,
                                                        operand2=op2_name,
                                                        op_width1=op1_bits,
                                                        op_width2=op2_bits))

                    new_operands_name.append(wire_name)
                    new_operands_bits.append(n_bits_addition)

                if len(operands_name) % 2 != 0:
                    new_operands_name.append(operands_name[-1])
                    new_operands_bits.append(operands_bits[-1])

                operands_name = new_operands_name
                operands_bits = new_operands_bits
                n_stage += 1

            module.add_assignment(Assignment("o", operands_name[0]))
            adder_modules.append(module)

        return adder_modules

    def _build_argmax_module(self):
        n_output_bits = self._bitwidth(self._n_classes - 1)

        module = HdlModule("argmax")
        module.add_port(Port("i", "in", self._n_classes * self._sum_bit_length))
        module.add_port(Port("o", "out", n_output_bits))

        if self._n_classes <= 1:
             module.add_assignment(Assignment("o", f"{n_output_bits}'d0"))
             return module

        for i in range(self._n_classes):
            sum_wire_name = f"sum_{i}"
            module.add_wire(Wire(sum_wire_name, self._sum_bit_length))
            module.add_assignment(Assignment(
                target=sum_wire_name,
                expression=f"i[{(i + 1) * self._sum_bit_length - 1}:{i * self._sum_bit_length}]"
            ))

        final_expr = f"{n_output_bits}'d{self._n_classes - 1}"
        for i in range(self._n_classes - 1):
            conditions = []
            for j in range(self._n_classes):
                if i == j: continue
                conditions.append(f"(sum_{i} >= sum_{j})")

            cond_str = " & ".join(conditions)
            final_expr = f"({cond_str}) ? {n_output_bits}'d{i} : ({final_expr})"

        module.add_assignment(Assignment("o", final_expr))
        return module

    def _build_quantization_module(self):
        module = HdlModule("quantization")
        module.add_port(Port("i", "in", self._n_features * self._bits_features))
        module.add_port(Port("o", "out", self._n_features * self._w_feature))

        max_out_val = 2**self._w_feature - 1

        for feature_idx in range(self._n_features):
            # Port slicing definitions
            input_msb = (feature_idx + 1) * self._bits_features - 1
            input_lsb = feature_idx * self._bits_features
            output_msb = (feature_idx + 1) * self._w_feature - 1
            output_lsb = feature_idx * self._w_feature

            input_signal = f"i[{input_msb}:{input_lsb}]"
            output_target = f"o[{output_msb}:{output_lsb}]"

            x_min = self._X_min[feature_idx]
            x_max = self._X_max[feature_idx]
            diff = x_max - x_min
            
            # --- Case 1: Constant (Min == Max) ---
            if diff == 0:
                zero_str = self._int_2_bitstring(0, self._w_feature)
                module.add_assignment(Assignment(output_target, zero_str))
                continue

            # Initialize with default value (mapped to Max)
            default_out_val = max_out_val
            final_expr = self._int_2_bitstring(default_out_val, self._w_feature)

            # --- Case 2: Gaps (Range < 7) ---
            if diff < max_out_val:
                # Direct mapping loop: Iterate backwards to build priority logic
                start_check = int(x_max) - 1
                end_check = int(x_min) - 1 
                
                for x_val in range(start_check, end_check, -1):
                    # Calculate exact output integer
                    raw_y = (x_val - x_min) / diff * max_out_val
                    y_int = int(round(raw_y))
                    
                    thresh_str = self._int_2_bitstring(x_val, self._bits_features)
                    out_str = self._int_2_bitstring(y_int, self._w_feature)
                    
                    # Wrap: (i <= x) ? y : (previous)
                    final_expr = f"({input_signal} <= {thresh_str}) ? {out_str} : ({final_expr})"

            # --- Case 3: Compression (Range >= 7) ---
            else:
                feature_thresholds = self._thresholds[feature_idx]
                
                # Iterate buckets backwards (6 down to 0)
                for k in range(max_out_val - 1, -1, -1):
                    cut_val = feature_thresholds[k]
                    bucket_out_val = k
                    
                    thresh_str = self._int_2_bitstring(int(cut_val), self._bits_features)
                    out_str = self._int_2_bitstring(bucket_out_val, self._w_feature)
                    
                    final_expr = f"({input_signal} <= {thresh_str}) ? {out_str} : ({final_expr})"

            module.add_assignment(Assignment(output_target, final_expr))

        return module
    
    def _build_myreg_module(self):
        module = HdlModule("myreg")
        module.add_parameter(Parameter("DataWidth", 16))

        module.add_port(Port("clk", "in", 1))
        module.add_port(Port("data_in", "in", "DataWidth"))
        module.add_port(Port("data_out", "out", "DataWidth", is_reg=True))

        verilog_code = (
            "always @(posedge clk) begin\n"
            "    data_out <= data_in;\n"
            "end"
        )

        vhdl_code = (
            "process(clk) begin\n"
            "    if rising_edge(clk) then\n"
            "        data_out <= data_in;\n"
            "    end if;\n"
            "end process;"
        )

        module.add_raw_code(RawCode(verilog_code, vhdl_code))
        return module

    def _build_one_comparator_module(self):        
        module = ComparatorModule("comparator_operator")

        module.add_port(Port("a", "in", self._w_feature))
        module.add_port(Port("b", "in", self._w_feature))
        module.add_port(Port("o", "out", 1))

        module.add_wire(Wire("r_diff", self._w_feature + 1))
        module.add_assignment(ArithmeticAssign(
            target="r_diff",
            ta_width=self._w_feature + 1,
            operand1="a",
            operand2="b",
            op_width1=self._w_feature,
            op_width2=self._w_feature,
            operation="-"))
        
        module.add_assignment(ConditionalAssign(
            target="o",
            condition=f"((r_diff != 0) & (r_diff[{self._w_feature}] == '0'))",
            true_val="1'b1",
            false_val="1'b0"
        ))

        return module
        
    def _build_comparator_modules(self):
        
        module = HdlModule("comparator")

        n_comparators = len(self._unique_features)
        module.add_port(Port("i", "in", self._n_features * self._w_feature))
        module.add_port(Port("o", "out", n_comparators))

        for i, (feature_index, threshold) in enumerate(self._unique_features):
            feature_input_bits = f"{(feature_index + 1) * self._w_feature - 1}:{feature_index * self._w_feature}"

            expr = (f"({{i[{feature_input_bits}]}} < "
                    f"({self._w_feature}'d{int(threshold)})) ? 1'b1 : 1'b0")

            # TODO - 2025-12-05 10:00:55: Change Assignment to ComparatorInstance
            module.add_assignment(ComparatorInstance(
                name=f"comparator_{i}",
                param_map={},
                port_map={
                    "a": f"i[{feature_input_bits}]",
                    "b": f"{self._w_feature}'d{int(threshold)}",
                    "o": f"o[{i}]"
                },
                target=f"o[{i}]",
                comparator=f"({{i[{feature_input_bits}]}})",
                threshold=f"({self._w_feature}'d{int(threshold)})"
            ))
        
        if self._pipeline[0] != 0:
            module.add_wire(Wire("binary_features_reg", n_comparators))
            module.add_instance(ModuleInstance(
                name="feature_reg",
                module_type="myreg",
                param_map={"DataWidth": n_comparators},
                port_map={
                    "clk": "clk",
                    "data_in": "binary_features",
                    "data_out": "binary_features_reg"
                }
            ))

        return module

    # ==============================================================
    # Helper Methods
    # ==============================================================

    def _link_parents(self, model_struct):
        """Helper to add parent links, needed for 'equation' style."""
        for tree in model_struct:
            for node_id, node in tree.items():
                if node['type'] == 'split':
                    yes_id = str(node.get('yes', -1))
                    no_id = str(node.get('no', -1))

                    if yes_id in tree:
                        tree[yes_id]['parent_node'] = node_id
                        tree[yes_id]['parent_yesno'] = 'yes'
                    if no_id in tree:
                        tree[no_id]['parent_node'] = node_id
                        tree[no_id]['parent_yesno'] = 'no'
        return model_struct

    def _calculate_sum_bit_length(self):
        if self._trees_bit_length is None:
            raise ValueError("Cannot calculate sum bit length before tree modules are built.")

        # max_sums = []
        # for i in range(self._n_classes):
        #     class_trees_bits = self._trees_bit_length[i, :]
        #     max_tree_vals = (np.power(2, class_trees_bits) - 1).sum()
        #     max_sums.append(self._classes_bias[i] + max_tree_vals)

        # self._sum_bit_length = self._bitwidth(np.max(max_sums))

        #NOTE - 2025-12-04 10:38:07: Change to calculate the sum bit length based
        # on the theoretical maximum value of each tree (sum of all n trees with
        # maximum value for their bit-width (self._w_tree)), plus the class bias.
        # e.g.: w_tree = 3 and 7 trees -> propagate max value of 7 (111) for each tree:
        # 111 + 111 + 111 + 111 + 111 + 111 + 111 = 7 * 7 = 49 -> bitwidth = 6
        max_sums = []
        for i in range(self._n_classes):
            max_tree_vals = (2**self._w_tree - 1) * self._n_trees_per_class
            max_sums.append(self._classes_bias[i] + max_tree_vals)
        self._sum_bit_length = self._bitwidth(np.max(max_sums))

    def _extract_unique_features(self):
        uf_list = []
        for tree in self._treelut_model_struct:
            uf_list.extend([(d['feature'], d['threshold']) for d in tree.values() if d['type'] == 'split'])
        uf_list = list(set(uf_list))
        uf_list.sort(key=lambda x: (x[0], x[1]))
        return uf_list

    def _set_style(self, style):
        if style not in ['mux', 'equation']:
            raise ValueError("Style must be either 'mux' or 'equation'.")
        return style

    def _bitwidth(self, N):
        if N >= 0:
            return int(np.floor(np.log2(N)) + 1) if N > 0 else 1
        else:
            return int(np.floor(np.log2(abs(N))) + 2) if N != -1 else 1

    def _get_thresholds(self):
        if self._X_min is None or self._X_max is None:
            raise ValueError("X_min and X_max are required.")

        n_cuts = 2**self._w_feature - 1
        thresholds = np.full((self._n_features, n_cuts), -1, dtype=int)
        max_out_val = n_cuts

        for i in range(self._n_features):
            min_v = self._X_min[i]
            max_v = self._X_max[i]
            diff = max_v - min_v

            # Skip Constant or Gap cases (handled directly in build module)
            if diff < max_out_val:
                continue

            # Compression Case: Calculate integer upper bounds for each bucket
            # Transition occurs mathematically at k + 0.5
            k_values = np.arange(n_cuts)
            boundary_values = k_values + 0.5
            
            # Map back to original scale
            real_boundaries = min_v + (boundary_values / max_out_val) * diff
            
            # Floor to get the integer inclusive upper bound
            thresholds[i, :] = np.floor(real_boundaries).astype(int)

        return thresholds

    def _int_2_bitstring(self, value, bits):
        if value < 0:
            value = (1 << bits) + value
        result = format(value, f'0{bits}b')
        return f"{bits}'b{result}"

#================================
# Verilog Code Generation Visitor
#================================

class VerilogVisitor:
    """
    Walks the HDL AST and generates Verilog code.
    """

    def _width_str_v(self, width):
        """Helper to format Verilog width strings."""
        if width == 1 or width is None:
            return ""
        if isinstance(width, int):
            return f"[{width-1}:0] "
        return f"[{width}-1:0] " # For strings like 'DataWidth'

    def visit_hdlmodule(self, mod: HdlModule) -> str:
        """Generates a Verilog module string."""

        # --- Parameters ---
        params = []
        if mod.parameters:
            for p in mod.parameters:
                params.append(p.accept(self))
            params_str = f"\n#(\n    {', '.join(params)}\n)\n"
        else:
            params_str = ""

        # --- Ports ---
        ports = []
        if mod.ports:
            for p in mod.ports:
                ports.append(p.accept(self))
            ports_str = f"(\n    {', '.join(ports)}\n);"
        else:
            ports_str = "();"

        # --- Body ---
        header = "\n".join([h.accept(self) for h in mod.headers])
        wires = "\n".join([w.accept(self) for w in mod.wires])
        assigns = "\n".join([a.accept(self) for a in mod.assignments])
        instances = "\n".join([i.accept(self) for i in mod.instances])
        raw_code = "\n".join([r.accept(self) for r in mod.raw_code_blocks])

        # --- Final Assembly ---
        return (
            f"{header}\n"
            f"module {mod.name} {params_str}{ports_str}\n\n"
            f"{wires}\n\n"
            f"{assigns}\n\n"
            f"{instances}\n\n"
            f"{raw_code}\n\n"
            f"endmodule\n"
        )

    def visit_parameter(self, param: Parameter) -> str:
        return f"parameter {param.name} = {param.value}"

    def visit_port(self, port: Port) -> str:
        direction = port.direction
        reg_str = "reg " if port.is_reg and direction == "out" else ""
        width = self._width_str_v(port.width)
        return f"{direction} {reg_str}{width}{port.name}"

    def visit_wire(self, wire: Wire) -> str:
        kind = "reg" if wire.is_reg else "wire"
        width = self._width_str_v(wire.width)
        return f"    {kind} {width}{wire.name};"

    def visit_assignment(self, assign: Assignment) -> str:
        return f"    assign {assign.target} = {assign.expression};"

    def visit_conditionalassign(self, mux: ConditionalAssign) -> str:
        return (
            f"    assign {mux.target} = {mux.condition} ? "
            f"{mux.true_val} : {mux.false_val};"
        )
    
    def visit_arithmeticassign(self, add: ArithmeticAssign) -> str:
        return (
            f"    assign {add.target} = {add.operand1} + {add.operand2};"
        )

    def visit_comparatorinstance(self, comp: ComparatorInstance) -> str:
        return (
            f"    assign {comp.target} = ({comp.comparator} < {comp.threshold}) ? 1'b1 : 1'b0;"
        )

    def visit_moduleinstance(self, inst: ModuleInstance) -> str:
        # --- Parameters ---
        params_str = ""
        if inst.param_map:
            params = [f".{k}({v})" for k, v in inst.param_map.items()]
            params_str = f"#(\n        {', '.join(params)}\n    ) "

        # --- Ports ---
        ports = [f".{k}({v})" for k, v in inst.port_map.items()]
        ports_str = f"(\n        {', '.join(ports)}\n    )"

        return (
            f"    {inst.module_type} {params_str}{inst.name} {ports_str};"
        )

    def visit_rawcode(self, raw: RawCode) -> str:
        # Indent the raw code to fit module body
        return "\n".join([f"    {line}" for line in raw.verilog_code.split('\n')])

    def visit_header(self, header: Header) -> str:
        return header.comment

#================================
# VHDL Code Generation Visitor
#================================

class VHDLVisitor:
    """
    Walks the HDL AST and generates VHDL code.

    Includes a preamble and basic Verilog-to-VHDL operator
    translation for simple assignments.
    """

    def _width_str_vhdl(self, width):
        """Helper to format VHDL type strings."""
        if width == 1 or width is None:
            return "std_logic"
        if isinstance(width, int):
            return f"std_logic_vector({width-1} downto 0)"
        return f"std_logic_vector({width}-1 downto 0)" # For strings

    def _preamble(self):
        """Returns the standard VHDL library preamble."""
        return (
            "library ieee;\n"
            "use ieee.std_logic_1164.all;\n"
            "use ieee.numeric_std.all;\n"
            "use ieee.std_logic_unsigned.all;\n\n"
        )

    def _int_2_bitstring(self, value, bits):
        #HACK - 2025-12-02 18:32:19: Temporary fix integer to bitstring for VHDL
        if value < 0:
            value = (1 << bits) + value
        result = format(value, f'0{bits}b')
        return f"{bits}'b{result}"

    def _translate_expr(self, expr: str) -> str:
        """
        Performs basic translation of Verilog expressions to VHDL.
        Now supports complex nested ternaries.
        """
        # 1. Basic Operator Replacement
        expr = expr.replace("~", "not ")
        expr = expr.replace("&", " and ")
        expr = expr.replace("|", " or ")
        expr = expr.replace("^", " xor ")
        expr = expr.replace("==", "=")
        expr = expr.replace("!=", "/=")
        expr = expr.replace("1'b1", "'1'")
        expr = expr.replace("1'b0", "'0'")
        expr = expr.replace("1'd1", "'1'")
        expr = expr.replace("1'd0", "'0'")
        
        # 2. Verilog 'd' format: 4'd10 -> bitsting "1010"
        expr = re.sub(r"\d+'d(\d+)", lambda m: self._int_2_bitstring(int(m.group(1)), int(m.group(0).split("'")[0])), expr)

        #2.1 Verilog 'b' format: 4'b1010 -> "1010"
        expr = re.sub(r"\d+'b([01_]+)", lambda m: f'"{m.group(1).replace("_", "")}"', expr)
        

        #3 Verilog single bit slicing: x[3:3] -> x(3)
        expr = re.sub(r"(\w+)\[(\d+):\2\]", r"\1(\2)", expr)

        # 3.1. Verilog bit slicing: x[3:0] -> x(3 downto 0)
        expr = re.sub(r"(\w+)\[(\d+):(\d+)\]", r"\1(\2 downto \3)", expr)

        # 4. Verilog indexing: x[3] -> x(3)
        expr = re.sub(r"(\w+)\[(\d+)\]", r"\1(\2)", expr)

        # 5. Handle Ternaries Recursively
        # This replaces the old simple regex
        if '?' in expr:
            expr = self._unwind_ternary(expr)
        
        return expr

    def _unwind_ternary(self, expr: str) -> str:
        """
        Recursively parses a Verilog ternary string (cond ? true : false)
        and converts it to VHDL (true when cond = '1' else false).
        Handles nested parentheses.
        """
        expr = expr.strip()
        
        # Remove surrounding parentheses if they wrap the entire expression
        # e.g., "(a ? b : c)" -> "a ? b : c"
        while expr.startswith('(') and expr.endswith(')'):
            # Verify these parens actually match each other and aren't just
            # two separate groups like "(a) and (b)"
            depth = 0
            is_wrapped = True
            for i, char in enumerate(expr[:-1]):
                if char == '(': depth += 1
                elif char == ')': depth -= 1
                if depth == 0:
                    is_wrapped = False
                    break
            
            if is_wrapped:
                expr = expr[1:-1].strip()
            else:
                break

        # Base case: If no ternary operator, return the value
        if '?' not in expr:
            return expr

        # Find the split points based on parenthesis depth
        depth = 0
        q_index = -1 # Index of '?'
        c_index = -1 # Index of ':'

        # 1. Find the condition (stop at first '?' at depth 0)
        for i, char in enumerate(expr):
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            elif char == '?' and depth == 0:
                q_index = i
                break
        
        # If we didn't find a top-level '?', it's likely inside parens that we
        # failed to strip, or logic error. Return as is.
        if q_index == -1:
            return expr

        # 2. Find the matching colon for this specific '?'
        # We start searching AFTER the '?'
        depth = 0
        for i in range(q_index + 1, len(expr)):
            char = expr[i]
            if char == '(': depth += 1
            elif char == ')': depth -= 1
            elif char == ':' and depth == 0:
                c_index = i
                break

        if c_index == -1:
            return expr # Malformed ternary

        # 3. Split the string
        cond_part = expr[:q_index].strip()
        true_part = expr[q_index+1 : c_index].strip()
        false_part = expr[c_index+1:].strip()

        # 4. Recursively process the true and false parts
        # (This handles the nested ternaries)
        true_vhdl = self._unwind_ternary(true_part)
        false_vhdl = self._unwind_ternary(false_part)

        # 5. Construct VHDL string
        # Check if the condition looks like a comparison, if so, don't add = '1'
        if any(op in cond_part for op in ["=", ">", "<", "and", "or"]):
             return f"{true_vhdl} when {cond_part} else\n {false_vhdl}"
        else:
             return f"{true_vhdl} when {cond_part} = '1' else\n {false_vhdl}"

    def visit_hdlmodule(self, mod: HdlModule) -> str:
        """Generates VHDL entity and architecture strings."""

        # --- 1. Entity ---
        params_list = [p.accept(self) for p in mod.parameters]
        ports_list = [p.accept(self) for p in mod.ports]

        # Generic block
        if params_list:
            generics_str = (
                "generic (\n"
                f"    {';\n    '.join(params_list)}\n"
                ");\n"
            )
        else:
            generics_str = ""

        # Port block
        if ports_list:
            ports_str = (
                "port (\n"
                f"    {';\n    '.join(ports_list)}\n"
                ");\n"
            )
        else:
            ports_str = ""

        entity_str = (
            f"entity {mod.name} is\n"
            f"{generics_str}"
            f"{ports_str}"
            f"end entity {mod.name};\n\n"
        )

        # --- 2. Architecture ---
        header = "\n".join([h.accept(self) for h in mod.headers])
        wires = "\n".join([w.accept(self) for w in mod.wires])
        assigns = "\n".join([a.accept(self) for a in mod.assignments])
        instances = "\n".join([i.accept(self) for i in mod.instances])
        raw_code = "\n".join([r.accept(self) for r in mod.raw_code_blocks])

        arch_str = (
            f"architecture rtl of {mod.name} is\n\n"
            f"{wires}\n\n"
            "begin\n\n"
            f"{assigns}\n\n"
            f"{instances}\n\n"
            f"{raw_code}\n\n"
            "end architecture rtl;\n"
        )

        return f"{header}\n\n" + self._preamble() + entity_str + arch_str

    def visit_comparatormodule(self, comp: ComparatorModule) -> str:
        
        """Generates VHDL for ComparatorModule."""
        # Reuse the HdlModule visitor
        return self.visit_hdlmodule(comp)

    def visit_parameter(self, param: Parameter) -> str:
        return f"{param.name} : integer := {param.value}"

    def visit_port(self, port: Port) -> str:
        # VHDL 'reg' is handled by process, not port declaration
        vhdl_dir = 'in' if port.direction == 'input wire' else 'out'
        vhdl_type = self._width_str_vhdl(port.width)
        return f"{port.name} : {vhdl_dir} {vhdl_type}"

    def visit_wire(self, wire: Wire) -> str:
        # 'reg' is implemented by 'RawCode' (process), here it's just a signal
        vhdl_type = self._width_str_vhdl(wire.width)
        return f"    signal {wire.name} : {vhdl_type};"

    def visit_assignment(self, assign: Assignment) -> str:
        expr = self._translate_expr(assign.expression)
        assign.target = self._translate_expr(assign.target)
        return f"    {assign.target} <= {expr};"

    def visit_arithmeticassign(self, add: ArithmeticAssign) -> str:

        if add.op_width1 < add.ta_width:
            diff = add.ta_width - add.op_width1
            # Shift right by adding zeros at MSB
            if diff == 1:
                op1 = f"'0' & {self._translate_expr(add.operand1)}"
            else:
                op1 = f'"{'0'* diff}"' + " & " + f"{self._translate_expr(add.operand1)}"
        else:
            op1 = add.operand1
        
        if add.op_width2 < add.ta_width:
            diff = add.ta_width - add.op_width2
            # Shift right by adding zeros at MSB
            if diff == 1:
                op2 = f"'0' & {self._translate_expr(add.operand2)}"
            else:
                op2 = f'"{'0'* diff}"' + " & " + f"{self._translate_expr(add.operand2)}"
        else:
            op2 = add.operand2

            
        return f"    {add.target} <= ({op1}) + ({op2});"

    def visit_conditionalassign(self, mux: ConditionalAssign) -> str:
        # VHDL conditional signal assignment
        cond = self._translate_expr(mux.condition)
        true_val = self._translate_expr(mux.true_val)
        false_val = self._translate_expr(mux.false_val)
        mux.target = self._translate_expr(mux.target)

        return (
            f"    {mux.target} <= {true_val} when {cond} = '1' else\n"
            f"                  {false_val};"
        )

    def visit_comparatorinstance(self, inst: ComparatorInstance) -> str:
        return self.visit_moduleinstance(inst)


    def visit_moduleinstance(self, inst: ModuleInstance) -> str:
        # --- Parameters (Generic Map) ---
        params_str = ""
        if inst.param_map:
            params = [f"{k} => {self._translate_expr(v)}" for k, v in inst.param_map.items()]
            params_str = (
                f"generic map (\n"
                f"        {',\n        '.join(params)}\n"
                f"    ) "
            )

        # --- Ports (Port Map) ---
        ports = [f"{k} => {self._translate_expr(v)}" for k, v in inst.port_map.items()]
        ports_str = (
            f"port map (\n"
            f"        {',\n        '.join(ports)}\n"
            f"    )"
        )

        return (
            f"    {inst.name} : entity work.{inst.module_type}(rtl)\n"
            f"    {params_str}{ports_str};"
        )

    def visit_rawcode(self, raw: RawCode) -> str:
        # Indent the raw VHDL code
        return "\n".join([f"    {line}" for line in raw.vhdl_code.split('\n')])

    def visit_header(self, header: Header) -> str:
        # For each line in the comment, prepend '-- '
        vhdl_comment = "\n".join([f"-- {line}" for line in header.comment.split('\n')])
        return vhdl_comment