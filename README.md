Introduction
PyAXEngine implements the Python API of Axera NPU Runtime based on the cffi module. Its Python API is highly compatible with ONNXRuntime and supports both development boards and M.2 computing cards, making it convenient for open source community developers to use Python scripts to quickly build NPU inference scripts.

Supported chips

AX650N
AX630C
Environment Version

python >= 3.8
cffi >= 1.0.0
ml-dtypes >= 0.1.0
numpy >= 1.22.0
Get started quickly
Demonstration based on the community development board Aixinpai Pro (AX650N)

Get the wheel package and install it
Download Link
Copy axengine-x.x.x-py3-none-any.whlto the development board and execute pip install axengine-x.x.x-py3-none-any.whlinstallation
Simple Example
Copy classification.py to the development board and execute it.

root@ax650:~/samples# python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg
[INFO] Available providers:  ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.10.1s
[INFO] Model type: 0 (single core)
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
  ------------------------------------------------------
  Top 5 Predictions:
    Class Index: 282, Score: 9.774
    Class Index: 278, Score: 8.981
    Class Index: 277, Score: 8.453
    Class Index: 281, Score: 8.321
    Class Index: 287, Score: 7.924
  ------------------------------------------------------
  min =   0.890 ms   max =   22.417 ms   avg =   1.119 ms
  ------------------------------------------------------
The example also demonstrates how to select a compute device: this means it can be run on development boards such as the AX650/AX630C , as well as on an AX650 M.2 computing card.

The way to switch computing devices is to -pspecify the parameter, such as -p AxEngineExecutionProvidermeans to use the NPU on the development board for inference, and -p AXCLRTExecutionProvidermeans to use the M.2 computing card for inference. Note: When using the M.2 computing card for inference, the computing card needs to be inserted into the host machine and the driver must have been installed, see: axcl for details .

root@ax650:~/samples# python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -p AXCLRTExecutionProvider
[INFO] Available providers:  ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
  ------------------------------------------------------
  Top 5 Predictions:
    Class Index: 282, Score: 9.774
    Class Index: 278, Score: 8.981
    Class Index: 277, Score: 8.453
    Class Index: 281, Score: 8.321
    Class Index: 287, Score: 7.924
  ------------------------------------------------------
  min =   1.587 ms   max =   12.624 ms   avg =   1.718 ms
  ------------------------------------------------------
root@ax650:~/samples# python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -p AxEngineExecutionProvider
[INFO] Available providers:  ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.10.1s
[INFO] Model type: 0 (single core)
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
  ------------------------------------------------------
  Top 5 Predictions:
    Class Index: 282, Score: 9.774
    Class Index: 278, Score: 8.981
    Class Index: 277, Score: 8.453
    Class Index: 281, Score: 8.321
    Class Index: 287, Score: 7.924
  ------------------------------------------------------
  min =   0.897 ms   max =   22.542 ms   avg =   1.125 ms
  ------------------------------------------------------
Community Contributors
zylo117 : Provides a cffi-based implementation of the AXCL Runtime Python API

Related Projects
ax-samples
ax-llm
pulsar2
axcl
Technical Discussion
Github issues
QQ group: 139953715
