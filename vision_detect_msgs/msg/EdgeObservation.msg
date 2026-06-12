bool edge_visible            # 是否检测到边界
string edge_side             # "front", "rear", "left", "right"
float32 d_edge_cm            # 到边界的距离 (cm)
string edge_type             # "cell_edge", "block_edge"
string safety_level          # "EDGE_CLEAR", "EDGE_CAUTION", "EDGE_STOP", "EDGE_FAULT"