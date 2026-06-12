std_msgs/Header header
string localization_state    # "NORMAL", "FAULT"
uint32 block_id              # 当前大板 ID
int32 cell_row               # 小板行号
int32 cell_col               # 小板列号
int32 inner_row              # 小板内部格子行索引
int32 inner_col              # 小板内部格子列索引