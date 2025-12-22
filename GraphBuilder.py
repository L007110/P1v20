# -*- coding: utf-8 -*-
import numpy as np
import torch
from collections import defaultdict
from logger import debug, debug_print, set_debug_mode
import Parameters
from Parameters import *
from ChannelModel import global_channel_model
from Parameters import V2V_CHANNEL_BANDWIDTH, TRANSMITTDE_POWER
import traceback
import sys

# 尝试导入新参数，如果不存在则使用默认值
try:
    from Parameters import GNN_INFERENCE_RADIUS
except ImportError:
    GNN_INFERENCE_RADIUS = 500.0


class GraphBuilder:
    """
    动态图构建器 (稳定修复版)
    """

    def __init__(self):
        self.edge_types = ['communication', 'interference', 'proximity']
        self.communication_threshold = 500.0
        self.interference_threshold = 300.0
        self.proximity_threshold = 200.0

        # 【关键修改 1】将特征维度增加到 12，确保容纳所有特征，不再截断
        self.rsu_feature_dim = 12
        self.vehicle_feature_dim = 6
        self.max_feature_dim = max(self.rsu_feature_dim, self.vehicle_feature_dim)
        self.comm_edge_feature_dim = 4
        debug("GraphBuilder initialized (Stable Version)")

    def build_dynamic_graph(self, dqn_list, vehicle_list, epoch):
        try:
            nodes = self._create_nodes(dqn_list, vehicle_list)
            edges = self._create_edges(nodes, dqn_list, vehicle_list, epoch)
            graph_data = {
                'nodes': nodes,
                'edges': edges,
                'node_features': self._extract_node_features(nodes, dqn_list, vehicle_list),
                'edge_features': self._extract_edge_features(edges, nodes),
                'metadata': {'epoch': epoch, 'num_rsu_nodes': len(dqn_list)}
            }
            return graph_data
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Graph Build Failed at Epoch {epoch}!")
            traceback.print_exc(file=sys.stdout)
            raise e

    def _create_nodes(self, dqn_list, vehicle_list):
        nodes = {'rsu_nodes': [], 'vehicle_nodes': []}
        for i, dqn in enumerate(dqn_list):
            nodes['rsu_nodes'].append({
                'id': f"rsu_{dqn.dqn_id}", 'type': 'rsu', 'original_id': dqn.dqn_id,
                'position': (dqn.bs_loc[0], dqn.bs_loc[1]),
                'features': self._extract_rsu_features(dqn)
            })
        for i, vehicle in enumerate(vehicle_list):
            nodes['vehicle_nodes'].append({
                'id': f"vehicle_{vehicle.id}", 'type': 'vehicle', 'original_id': vehicle.id,
                'position': vehicle.curr_loc, 'direction': vehicle.curr_dir,
                'features': self._extract_vehicle_features(vehicle)
            })
        return nodes

    def _extract_rsu_features(self, dqn):
        # 1. 基础特征 (5)
        vehicle_count = len(dqn.vehicle_in_dqn_range_by_distance) if hasattr(dqn,
                                                                             'vehicle_in_dqn_range_by_distance') else 0
        features = [
            dqn.bs_loc[0] / SCENE_SCALE_X,
            dqn.bs_loc[1] / SCENE_SCALE_Y,
            float(getattr(dqn, 'vehicle_exist_curr', False)),
            vehicle_count / 10.0,
            getattr(dqn, 'prev_snr', 0.0) / 50.0,
        ]

        # 2. CSI 特征 (2)
        csi_distance, csi_snr = 0.0, 0.0
        if USE_UMI_NLOS_MODEL and hasattr(dqn, 'csi_states_curr') and dqn.csi_states_curr:
            csi_distance = dqn.csi_states_curr[0] / 1000.0 if len(dqn.csi_states_curr) > 0 else 0.0
            csi_snr = dqn.csi_states_curr[3] / 50.0 if len(dqn.csi_states_curr) > 3 else 0.0
        features.extend([csi_distance, csi_snr])

        # 3. 干扰特征 (2)
        v2i_int = getattr(dqn, 'prev_v2i_interference', 0.0)
        v2v_int = getattr(dqn, 'prev_v2v_interference', 0.0)
        features.append(np.clip(v2i_int / 1e-9, 0.0, 1.0))
        features.append(np.clip(v2v_int / 1e-9, 0.0, 1.0))

        # 4. 方向特征 (2) - 【关键修改】不再依赖 hasattr，强制计算，没有就填 0
        dir_x, dir_y = 0.0, 0.0
        try:
            target_rx = Parameters.V2I_LINK_POSITIONS[0]['rx']
            curr_pos = (dqn.bs_loc[0], dqn.bs_loc[1])
            if hasattr(dqn, 'vehicle_in_dqn_range_by_distance') and dqn.vehicle_in_dqn_range_by_distance:
                if len(dqn.vehicle_in_dqn_range_by_distance) > 0:
                    curr_pos = dqn.vehicle_in_dqn_range_by_distance[0].curr_loc

            dx = target_rx[0] - curr_pos[0]
            dy = target_rx[1] - curr_pos[1]
            dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
            dir_x = dx / dist
            dir_y = dy / dist
        except:
            # 如果 Parameters 还没准备好，或者没有链接，默认方向为 0
            dir_x, dir_y = 0.0, 0.0

        features.append(dir_x)
        features.append(dir_y)

        # 5. 补齐位 (1) - 之前是重复添加 v2v，为了对齐维度我们加上它
        features.append(np.clip(v2v_int / 1e-9, 0.0, 1.0))

        # 6. 维度强制对齐
        # 现在的 features 长度应该是 12。我们强制对齐到 self.rsu_feature_dim (12)
        if len(features) < self.rsu_feature_dim:
            features.extend([0.0] * (self.rsu_feature_dim - len(features)))
        else:
            features = features[:self.rsu_feature_dim]

        return features

    def _extract_vehicle_features(self, vehicle):
        features = [
            vehicle.curr_loc[0] / SCENE_SCALE_X,
            vehicle.curr_loc[1] / SCENE_SCALE_Y,
            (vehicle.curr_dir[0] + 1) / 2.0,
            (vehicle.curr_dir[1] + 1) / 2.0,
            float(vehicle.first_occur),
        ]
        dist = vehicle.distance_to_bs / 1000.0 if (
                    hasattr(vehicle, 'distance_to_bs') and vehicle.distance_to_bs is not None) else 0.0
        features.append(dist)

        if len(features) < self.vehicle_feature_dim:
            features.extend([0.0] * (self.vehicle_feature_dim - len(features)))
        else:
            features = features[:self.vehicle_feature_dim]
        return features

    def _create_edges(self, nodes, dqn_list, vehicle_list, epoch):
        edges = {
            'communication': self._calculate_communication_edges(nodes, dqn_list, vehicle_list),
            'interference': [],
            'proximity': self._calculate_proximity_edges(nodes, dqn_list, vehicle_list)
        }
        return edges

    def _calculate_communication_edges(self, nodes, dqn_list, vehicle_list):
        communication_edges = []
        for rsu_node in nodes['rsu_nodes']:
            rsu_dqn = next((d for d in dqn_list if d.dqn_id == rsu_node['original_id']), None)
            if not rsu_dqn: continue

            for vehicle_node in nodes['vehicle_nodes']:
                vehicle = next((v for v in vehicle_list if v.id == vehicle_node['original_id']), None)
                if not vehicle: continue

                if (rsu_dqn.start[0] <= vehicle.curr_loc[0] <= rsu_dqn.end[0] and
                        rsu_dqn.start[1] <= vehicle.curr_loc[1] <= rsu_dqn.end[1]):
                    try:
                        distance = global_channel_model.calculate_3d_distance(
                            (rsu_dqn.bs_loc[0], rsu_dqn.bs_loc[1]), vehicle.curr_loc)

                        base_power = TRANSMITTDE_POWER * 0.1
                        csi_info = global_channel_model.get_channel_state_info(
                            (rsu_dqn.bs_loc[0], rsu_dqn.bs_loc[1]), vehicle.curr_loc,
                            tx_power=base_power, bandwidth=V2V_CHANNEL_BANDWIDTH
                        )
                        features = [
                            1.0 - (distance / self.communication_threshold),
                            distance / 1000.0,
                            csi_info['path_loss_total_db'] / 100.0,
                            csi_info['snr_db'] / 20.0
                        ]
                        if distance <= self.communication_threshold:
                            communication_edges.append({
                                'source': rsu_node['id'], 'target': vehicle_node['id'],
                                'type': 'communication', 'distance': distance, 'features': features
                            })
                    except:
                        pass
        return communication_edges

    def _calculate_proximity_edges(self, nodes, dqn_list, vehicle_list):
        proximity_edges = []
        all_nodes = nodes['rsu_nodes'] + nodes['vehicle_nodes']
        for i, node_i in enumerate(all_nodes):
            for j, node_j in enumerate(all_nodes):
                if i >= j: continue
                dist = np.sqrt((node_i['position'][0] - node_j['position'][0]) ** 2 + (
                            node_i['position'][1] - node_j['position'][1]) ** 2)
                if dist <= self.proximity_threshold:
                    proximity_edges.append({
                        'source': node_i['id'], 'target': node_j['id'],
                        'type': 'proximity', 'distance': dist, 'weight': 1.0 - (dist / self.proximity_threshold)
                    })
        return proximity_edges

    def _extract_node_features(self, nodes, dqn_list, vehicle_list):
        all_features = []
        node_types = []
        for rsu_node in nodes['rsu_nodes']:
            all_features.append(rsu_node['features'])
            node_types.append(0)
        for vehicle_node in nodes['vehicle_nodes']:
            all_features.append(vehicle_node['features'])
            node_types.append(1)

        feature_lengths = [len(f) for f in all_features]
        max_len = max(feature_lengths) if feature_lengths else 0
        for i in range(len(all_features)):
            if len(all_features[i]) < max_len:
                all_features[i].extend([0.0] * (max_len - len(all_features[i])))

        return {'features': torch.FloatTensor(all_features), 'types': torch.LongTensor(node_types)}

    def _extract_edge_features(self, edges, nodes):
        edge_features = {}
        node_id_to_index = {node['id']: idx for idx, node in enumerate(nodes['rsu_nodes'] + nodes['vehicle_nodes'])}

        for edge_type in self.edge_types:
            edge_list = edges[edge_type]
            if not edge_list:
                edge_features[edge_type] = None
                continue

            edge_indices = []
            edge_attrs = []
            for edge in edge_list:
                edge_indices.append([node_id_to_index[edge['source']], node_id_to_index[edge['target']]])
                if edge_type == 'communication':
                    edge_attrs.append(edge['features'])
                else:
                    edge_attrs.append([edge['weight']] + [0.0] * (self.comm_edge_feature_dim - 1))

            edge_features[edge_type] = {
                'edge_index': torch.LongTensor(edge_indices).t().contiguous(),
                'edge_attr': torch.FloatTensor(edge_attrs)
            }
        return edge_features

    def build_spatial_subgraph(self, center_dqn, all_dqns, all_vehicles, epoch, radius=GNN_INFERENCE_RADIUS):
        center_pos = center_dqn.bs_loc
        filtered_dqns = [d for d in all_dqns if d.dqn_id == center_dqn.dqn_id or np.sqrt(
            (d.bs_loc[0] - center_pos[0]) ** 2 + (d.bs_loc[1] - center_pos[1]) ** 2) <= radius]
        filtered_vehicles = [v for v in all_vehicles if np.sqrt(
            (v.curr_loc[0] - center_pos[0]) ** 2 + (v.curr_loc[1] - center_pos[1]) ** 2) <= radius]
        return self.build_dynamic_graph(filtered_dqns, filtered_vehicles, epoch)


global_graph_builder = GraphBuilder()