# Every kind of field a .msg can have, so week05's CDR codec can be checked
# against rclpy byte for byte.
int32 ANSWER=42
string GREETING="hi # not a comment"

bool b
byte o
char c
int8 i8
uint8 u8
int16 i16
uint16 u16
int32 i32
uint32 u32
int64 i64
uint64 u64
float32 f32
float64 f64
string s
wstring w
string<=8 bounded_s
int32[3] fixed_i32
float64[2] fixed_f64
uint8[4] fixed_u8
string[2] fixed_s
int16[] seq_i16
uint8[] seq_u8
string[] seq_s
wstring[] seq_w
float32[<=4] bounded_f32
geometry_msgs/Point p
geometry_msgs/Point[] points
geometry_msgs/Point[2] fixed_points
builtin_interfaces/Time[<=2] times
int32 with_default 7
float64[] seq_default [1.5, -2.0]
