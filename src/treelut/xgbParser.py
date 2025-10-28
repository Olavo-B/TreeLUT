# /*****************************************************************************/
#  * File: boostParser.py
#  * Author: Olavo Alves Barros Silva
#  * Contact: olavo.barros@ufv.com
#  * Date: 2025-10-06
#  * License: [License Type]
#  * Description: This module provides parsers for XGBoost and LightGBM boosting
#  * models using the Factory Pattern for automatic model type detection.
# /*****************************************************************************/

import re
import json
import logging
import numpy as np
import xgboost as xgb
import lightgbm as lgb # Import lightgbm
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class NullLogger:
    """A logger implementation that ignores all messages.

    This allows code to unconditionally call logger.info/debug/error without
    checking for None. Methods accept the same signature as SoraLogger but do
    nothing.
    """
    def debug(self, message: str, **kwargs):
        return None

    def info(self, message: str, **kwargs):
        return None

    def warning(self, message: str, **kwargs):
        return None

    def error(self, message: str, **kwargs):
        return None

    def critical(self, message: str, **kwargs):
        return None


# Singleton instance for convenience
NULL_LOGGER = NullLogger()



class BaseBoostParser(ABC):
    """Abstract base class for gradient boosting model parsers."""

    def __init__(self, model, logger = None ):
        # Initialize logger if not provided
        if logger is None:
            # Use NULL_LOGGER by default to avoid side-effects (file creation)
            logger = NULL_LOGGER
        self.logger = logger


        self._model = model
        self._features = None
        self._forest_model = None

        self._aux_bias = 0.0  # 2025-10-10 08:13:55: Placeholder for auxiliary bias if needed

    #=================#
    # Properties      #
    #=================#

    @property
    @abstractmethod
    def numpy_forest(self):
        """Convert forest to numpy format."""
        pass

    @property
    def nodes(self):
        """Get all nodes from all trees."""
        return [node for tree in self._forest_model for node in tree.values()]

    @property
    def trees(self):
        """Get all trees in the forest."""
        return self._forest_model

    @property
    @abstractmethod
    def base_score(self):
        """Get the base score of the model."""
        pass

    @property
    def features(self):
        """Get feature names."""
        return self._features


    @property
    def score_matrix(self):
        """
        Get leaf scores in a padded matrix form (trees x max_leaves).
        Shorter trees are padded with np.nan.
        """
        score_lists = []
        max_leaves = 0
        for tree in self._forest_model:
            leaf_scores = [node['value'] for node in tree.values() if node['type'] == 'leaf']
            score_lists.append(leaf_scores)
            if len(leaf_scores) > max_leaves:
                max_leaves = len(leaf_scores)

        # Now create the final padded matrix
        score_matrix = np.full((len(score_lists), max_leaves), np.nan)
        for i, scores in enumerate(score_lists):
            score_matrix[i, :len(scores)] = scores

        return score_matrix


    @score_matrix.setter
    def score_matrix(self, new_scores: Union[np.ndarray, List[List[float]]]):
        """Set new scores for leaf nodes from a matrix, ignoring np.nan padding."""
        self.logger.info("Setting new scores from score matrix")

        if len(new_scores) != len(self._forest_model):
            raise ValueError("Number of trees in new_scores does not match the model")

        for tree_idx, (tree, scores_row) in enumerate(zip(self._forest_model, new_scores)):
            # Filter out NaN values which are used for padding
            clean_scores = [s for s in scores_row if not np.isnan(s)]

            leaf_nodes = [node for node in tree.values() if node['type'] == 'leaf']

            # Validate against the count of clean scores
            if len(clean_scores) != len(leaf_nodes):
                raise ValueError(
                    f"Number of non-NaN scores for tree {tree_idx} ({len(clean_scores)}) "
                    f"does not match number of leaf nodes ({len(leaf_nodes)})"
                )

            for node, score in zip(leaf_nodes, clean_scores):
                node['value'] = score

    @property
    def score(self):
        """Get all leaf scores."""
        scores = [node['value'] for tree in self._forest_model
                 for node in tree.values() if node['type'] == 'leaf']
        return np.array(scores)

    @score.setter
    def score(self, new_score: List):
        """Set new scores for leaf nodes."""
        self.logger.info("Setting new scores for leaf nodes")

        score_list = list(new_score) if not isinstance(new_score, list) else new_score
        for tree_idx, tree in enumerate(self._forest_model):
            for node_idx, node in enumerate(tree.values()):
                if node['type'] == 'leaf':
                    if not score_list:
                        raise ValueError("Not enough scores provided for all leaf nodes")
                    node['value'] = score_list.pop(0)

    @property
    def n_features(self):
        """Get number of features."""
        return len(self._features)

    @property
    def n_trees(self):
        """Get number of trees."""
        return len(self._forest_model)

    @property
    def n_nodes(self):
        """Get total number of nodes."""
        return sum(len(tree) for tree in self._forest_model)

    @property
    def n_leaves(self):
        """Get total number of leaf nodes."""
        return sum(1 for tree in self._forest_model
                  for node in tree.values() if node['type'] == 'leaf')

    #=================#
    # Abstract Methods#
    #=================#

    @abstractmethod
    def _format_model(self):
        """Format the model into structured tree format."""
        pass

    @abstractmethod
    def _convert_to_numpy(self):
        """Convert the parsed forest to numpy arrays."""
        pass

    #=================#
    # Public Methods  #
    #=================#

    def get_parsed_model(self):
        """Return the parsed forest model."""
        self.logger.info("Returning parsed forest model")
        return self._forest_model

    def add_aux_bias(self, bias: float):
        """Add auxiliary bias to all leaf node values."""
        self.logger.debug(f"Adding auxiliary bias {bias} to all leaf nodes, current bias: {self._aux_bias}")
        self._aux_bias += bias


class XGBoostRegressionParser(BaseBoostParser):
    """Parser for XGBoost models (Regression and Classification)."""

    def __init__(self, xgb_model, logger = None):
        super().__init__(xgb_model, logger)

        self.logger.info("Initializing XGBoost Parser")


        self._features = xgb_model.get_booster().feature_names

        if not self._features:
            self._features = [f'feature_{i}' for i in range(xgb_model.n_features_in_)]
            self.logger.warning(f"No feature names found, generated {len(self._features)} default names")
        else:
            self.logger.info(f"Found {len(self._features)} feature names in model")

        self._format_model()

        self.logger.info("XGBoost Parser initialized successfully",
                        extra={"n_features": self.n_features, "n_trees": self.n_trees, "n_nodes": self.n_nodes})

    #=================#
    # Properties      #
    #=================#

    @property
    def numpy_forest(self):
        self.logger.info("Converting forest to numpy format")
        return self._convert_to_numpy()

    @property
    def base_score(self):
        config_str = self._model.get_booster().save_config()
        config = json.loads(config_str)
        base_score = float(config['learner']['learner_model_param']['base_score'])
        self.logger.debug(f"Base score retrieved: {base_score} and adjusted with aux_bias: {self._aux_bias}")
        return base_score + self._aux_bias

    #=================#
    # Private Methods #
    #=================#

    def __parse_node_line(self, line: str) -> dict:
        """Parse a split node line from XGBoost tree dump."""
        
        # --- MODIFIED ---
        # Changed (\w+) to ([^\<]+) to allow hyphens in feature names
        pattern = r'(\d+):\[([^\<]+)<(.+?)\] yes=(\d+),no=(\d+),missing=(\d+)'
        match = re.match(pattern, line.strip())

        if not match:
            # --- MODIFIED ---
            # Also changed (\w+) to ([^\<]+) here
            pattern_no_missing = r'(\d+):\[([^\<]+)<(.+?)\] yes=(\d+),no=(\d+)'
            match = re.match(pattern_no_missing, line.strip())
            if not match:
                error_msg = f"Could not parse split node line: {line}"
                self.logger.error(error_msg, module="XGBoostRegressionParser")
                raise ValueError(error_msg)
            node_id, feature, threshold, yes_child, no_child = match.groups()
            missing_child = no_child
        else:
            node_id, feature, threshold, yes_child, no_child, missing_child = match.groups()

        return {
            'index': node_id,
            'feature': feature,
            'threshold': float(threshold),
            'yes_value': yes_child,
            'no_value': no_child,
            'missing_value': missing_child
        }

    def __parse_leaf_line(self, line: str) -> tuple:
        """Parse a leaf node line from XGBoost tree dump."""
        match = re.match(r'(\d+):leaf=([-\d.e]+)', line.strip())
        if not match:
            error_msg = f"Could not parse leaf node line: {line}"
            self.logger.error(error_msg, module="XGBoostRegressionParser")
            raise ValueError(error_msg)

        node_id, value = match.groups()
        return node_id, float(value)

    def _format_model(self):
        """Format the XGBoost model into structured tree format."""

        formatted_trees = []
        global_id = 0
        model_dump = self._model.get_booster().get_dump()


        for tree_idx, tree_str in enumerate(model_dump):

            tree_lines = tree_str.strip().split('\n')
            nodes = {}

            for line in tree_lines:
                if 'leaf' in line:
                    node_id, value = self.__parse_leaf_line(line)
                    nodes[node_id] = {'type': 'leaf', 'value': float(value), 'global_id': global_id}
                else:
                    line_info = self.__parse_node_line(line)
                    try:
                        feature_idx = self._features.index(line_info['feature'])
                    except ValueError:
                        feature_idx = int(re.sub(r'\D', '', line_info['feature']))

                    nodes[line_info['index']] = {
                        'type': 'split',
                        'feature': feature_idx,
                        'threshold': line_info['threshold'],
                        'no': line_info['no_value'],
                        'yes': line_info['yes_value'],
                        'missing': line_info['missing_value'],
                        'global_id': global_id
                    }
                global_id += 1

            formatted_trees.append(nodes)


        self._forest_model = formatted_trees
        self.logger.info(f"Model formatting completed. Total nodes: {global_id}")

    def _convert_to_numpy(self):
        """Convert the parsed forest to numpy arrays."""
        if self._forest_model is None:
            error_msg = "Model has not been parsed yet."
            self.logger.error(error_msg, module="XGBoostRegressionParser")
            raise ValueError(error_msg)

        # Constants for property indices
        FEAT_IDX, THRESH_IDX, LEFT_IDX, RIGHT_IDX, VAL_IDX, GID_IDX, IS_LEAF_IDX, NODE_IDX = 0, 1, 2, 3, 4, 5, 6, 7

        numpy_trees = []
        total_nodes = 0

        for tree_idx, tree in enumerate(self._forest_model):
            n_nodes = len(tree)
            total_nodes += n_nodes
            numpy_tree = np.zeros((n_nodes, 8), dtype=np.float32)

            for node_id, node in tree.items():
                numpy_tree[int(node_id), FEAT_IDX] = float(node.get('feature', -1))
                numpy_tree[int(node_id), THRESH_IDX] = float(node.get('threshold', 0.0))
                numpy_tree[int(node_id), LEFT_IDX] = node.get('yes', -1)
                numpy_tree[int(node_id), RIGHT_IDX] = node.get('no', -1)
                numpy_tree[int(node_id), VAL_IDX] = float(node.get('value', 0.0))
                numpy_tree[int(node_id), GID_IDX] = float(node.get('global_id', -1))
                numpy_tree[int(node_id), IS_LEAF_IDX] = float(1 if node['type'] == 'leaf' else 0)
                numpy_tree[int(node_id), NODE_IDX] = node_id

            numpy_trees.append(numpy_tree)

        # Save all numpy_trees in a .txt file for debugging
        try:
            with open('numpy_trees_xgb.txt', 'w') as f:
                for i, tree in enumerate(numpy_trees):
                    f.write(f'Tree {i}:\n')
                    np.savetxt(f, tree, fmt='%.6f')
                    f.write('\n')
            self.logger.info("Numpy trees saved to 'numpy_trees_xgb.txt' for debugging")
        except Exception as e:
            self.logger.warning(f"Failed to save numpy trees debug file: {e}")


        return numpy_trees


class LightGBMRegressionParser(BaseBoostParser):
    """Parser for LightGBM models (Regression and Classification)."""

    def __init__(self, lgb_model, logger = None):
        super().__init__(lgb_model, logger)

        self.logger.info("Initializing LightGBM Parser")

        # --- MODIFIED ---
        # Validate model type
        if not isinstance(lgb_model, (lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.Booster)):
            error_msg = "The model must be an instance of lightgbm.LGBMRegressor, LGBMClassifier, or Booster."
            self.logger.error(error_msg)
            raise TypeError(error_msg)
        # --- END MODIFIED ---

        self._features = lgb_model.feature_name_

        if not self._features:
            self._features = [f'feature_{i}' for i in range(lgb_model.n_features_in_)]
            self.logger.warning(f"No feature names found, generated {len(self._features)} default names")
        else:
            self.logger.info(f"Found {len(self._features)} feature names in model")

        self._format_model()

        self.logger.info("LightGBM Parser initialized successfully",
                        extra = {"n_features": self.n_features, "n_trees": self.n_trees, "n_nodes": self.n_nodes})

    #=================#
    # Properties      #
    #=================#

    @property
    def numpy_forest(self):
        return self._convert_to_numpy()

    @property
    def base_score(self):
        """LightGBM uses 0.0 as base score by default."""
        return 0.0 + self._aux_bias

    #=================#
    # Private Methods #
    #=================#

    def __parse_tree_dict(self, tree_dict, tree_shrinkage=1.0, node_id=0, global_id_counter=None):
        """
        Recursively parse LightGBM tree dictionary structure.

        Args:
            tree_dict: Tree structure from LightGBM
            tree_shrinkage: Learning rate for this tree
            node_id: Current node ID in binary tree indexing
            global_id_counter: Counter for global node IDs
        """
        if global_id_counter is None:
            global_id_counter = {'count': 0}

        nodes = {}

        # Check if this is a leaf node
        if 'leaf_value' in tree_dict:
            # Apply shrinkage to leaf value
            leaf_value = float(tree_dict['leaf_value']) * tree_shrinkage

            nodes[str(node_id)] = {
                'type': 'leaf',
                'value': leaf_value,
                'global_id': global_id_counter['count'],
                'weight': tree_dict.get('leaf_weight', 0),
                'count': tree_dict.get('leaf_count', 0)
            }
            global_id_counter['count'] += 1
        else:
            # This is a split node
            feature_idx = tree_dict['split_feature']
            threshold = tree_dict['threshold']
            default_left = tree_dict.get('default_left', True)

            left_child_id = 2 * node_id + 1
            right_child_id = 2 * node_id + 2

            nodes[str(node_id)] = {
                'type': 'split',
                'feature': feature_idx,
                'threshold': float(threshold),
                'yes': str(left_child_id),
                'no': str(right_child_id),
                'missing': str(left_child_id if default_left else right_child_id),
                'global_id': global_id_counter['count'],
                'split_gain': tree_dict.get('split_gain', 0.0)
            }
            global_id_counter['count'] += 1

            # Recursively parse left and right children
            if 'left_child' in tree_dict:
                left_nodes = self.__parse_tree_dict(
                    tree_dict['left_child'],
                    tree_shrinkage,
                    left_child_id,
                    global_id_counter
                )
                nodes.update(left_nodes)

            if 'right_child' in tree_dict:
                right_nodes = self.__parse_tree_dict(
                    tree_dict['right_child'],
                    tree_shrinkage,
                    right_child_id,
                    global_id_counter
                )
                nodes.update(right_nodes)

        return nodes

    def _format_model(self):
        """Format the LightGBM model into structured tree format."""


        formatted_trees = []
        global_id = 0

        # Get model dump as dictionary
        model_dict = self._model.booster_.dump_model()
        tree_info = model_dict['tree_info']


        for tree_idx, tree in enumerate(tree_info):


            tree_structure = tree['tree_structure']

            global_id_counter = {'count': global_id}
            nodes = self.__parse_tree_dict(
                tree_structure,
                node_id=0,
                global_id_counter=global_id_counter
            )
            global_id = global_id_counter['count']
            formatted_trees.append(nodes)


        self._forest_model = formatted_trees
        self.logger.info(f"Model formatting completed. Total nodes: {global_id}")

    def _convert_to_numpy(self):
        """Convert the parsed forest to numpy arrays using sequential indexing."""
        if self._forest_model is None:
            error_msg = "Model has not been parsed yet."
            self.logger.error(error_msg, module="LightGBMRegressionParser")
            raise ValueError(error_msg)

        # Constants for property indices
        FEAT_IDX, THRESH_IDX, LEFT_IDX, RIGHT_IDX, VAL_IDX, GID_IDX, IS_LEAF_IDX = 0, 1, 2, 3, 4, 5, 6

        numpy_trees = []
        total_nodes = 0

        for tree_idx, tree in enumerate(self._forest_model):
            # Create mapping from string node IDs to sequential indices
            node_ids = sorted([int(k) for k in tree.keys()])
            id_to_idx = {str(node_id): idx for idx, node_id in enumerate(node_ids)}

            n_nodes = len(tree)
            total_nodes += n_nodes
            numpy_tree = np.zeros((n_nodes, 7), dtype=np.float32)

            for node_id_str, node in tree.items():
                idx = id_to_idx[node_id_str]

                numpy_tree[idx, FEAT_IDX] = float(node.get('feature', -1))
                numpy_tree[idx, THRESH_IDX] = float(node.get('threshold', 0.0))

                # Map child IDs to sequential indices
                if 'yes' in node:
                    numpy_tree[idx, LEFT_IDX] = float(id_to_idx.get(node['yes'], -1))
                else:
                    numpy_tree[idx, LEFT_IDX] = -1

                if 'no' in node:
                    numpy_tree[idx, RIGHT_IDX] = float(id_to_idx.get(node['no'], -1))
                else:
                    numpy_tree[idx, RIGHT_IDX] = -1

                numpy_tree[idx, VAL_IDX] = float(node.get('value', 0.0))
                numpy_tree[idx, GID_IDX] = float(node.get('global_id', -1))
                numpy_tree[idx, IS_LEAF_IDX] = float(1 if node['type'] == 'leaf' else 0)

            numpy_trees.append(numpy_tree)


        # Save for debugging
        try:
            with open('numpy_trees_lgbm.txt', 'w') as f:
                for i, tree in enumerate(numpy_trees):
                    f.write(f'Tree {i}:\n')
                    f.write(f'Shape: {tree.shape}\n')
                    np.savetxt(f, tree, fmt='%.6f',
                              header='Feature Threshold Left Right Value GlobalID IsLeaf')
                    f.write('\n')
            self.logger.info("Numpy trees saved to 'numpy_trees_lgbm.txt' for debugging")
        except Exception as e:
            self.logger.warning(f"Failed to save numpy trees debug file: {e}")


        return numpy_trees

    #=================#
    # Public Methods  #
    #=================#

    def get_leaf_node_ids(self, X):
        """
        Get the node IDs of leaves for each sample and tree.

        Args:
            X: Input features (numpy array or list)

        Returns:
            2D array where result[i][j] is the node ID string for sample i in tree j
        """

        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        leaf_node_ids = []

        for sample_idx in range(X.shape[0]):
            sample_leaves = []
            for tree in self._forest_model:
                node_id = '0'  # Start at root

                while tree[node_id]['type'] != 'leaf':
                    node = tree[node_id]
                    feature_val = X[sample_idx, node['feature']]

                    # LightGBM: left child (<= threshold), right child (> threshold)
                    if feature_val <= node['threshold']:
                        node_id = node['yes']
                    else:
                        node_id = node['no']

                sample_leaves.append(node_id)
            leaf_node_ids.append(sample_leaves)

        return leaf_node_ids


# /*****************************************************************************/
# FACTORY PATTERN IMPLEMENTATION
# /*****************************************************************************/

class BoostRegressionParser:
    """
    Factory class for creating appropriate boost model parsers.

    This class automatically detects the model type (XGBoost or LightGBM)
    and returns the appropriate parser instance.

    Usage:
        parser = BoostRegressionParser(model, logger)
        trees = parser.trees
        numpy_forest = parser.numpy_forest
    """

    def __new__(cls, model, logger = None) -> Union[XGBoostRegressionParser, LightGBMRegressionParser]:
        """
        Create and return the appropriate parser based on model type.

        Args:
            model: XGBoost or LightGBM model instance
            logger: Optional SoraLogger instance

        Returns:
            Instance of XGBoostRegressionParser or LightGBMRegressionParser

        Raises:
            TypeError: If model type is not supported
        """
        # Initialize logger if not provided
        if logger is None:
            logger = NULL_LOGGER

        logger.info("Detecting model type for parser creation",
                   module="BoostRegressionParser", function="__new__")

        # Check if model is XGBoost
        if isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier, xgb.Booster)):
            logger.info("Detected XGBoost model, creating XGBoostRegressionParser")
            return XGBoostRegressionParser(model, logger)

        # Check if model is LightGBM
        elif isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.Booster)):
            logger.info("Detected LightGBM model, creating LightGBMRegressionParser")
            return LightGBMRegressionParser(model, logger)

        # Unsupported model type
        else:
            model_type = type(model).__name__
            error_msg = (f"Unsupported model type: {model_type}. "
                        f"Supported types: XGBoost (XGBRegressor, XGBClassifier, Booster) "
                        f"and LightGBM (LGBMRegressor, LGBMClassifier, Booster)")
            logger.error(error_msg)
            raise TypeError(error_msg)
