from sparseecho import SparseEchoEngine, compile_query_plan
from sparseecho.io import open_capture_directory

capture = open_capture_directory("capture-demo")
engine = SparseEchoEngine(plan=compile_query_plan())
result = engine.process_capture(capture.slots, capture.valid)
print(result.identities)
