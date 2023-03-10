# from openvino.inference_engine import IECore
import os
import argparse
import gradio as gr 
import numpy as np 
import cv2
import onnx 
import onnxruntime as ort

# import PIL 

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX model to OpenVINO IR')
    parser.add_argument('--model',  type= str, required = True ,help='Path to ONNX model')
    parser.add_argument('--device', type= str, default = 'CPU', help='device to use, cpu or tpu')
    return parser.parse_args()

def cnn_classifier(image_path):
    """
    Process inference for rondelles
    Args:
        - image
    Returns:
        - segmentation mask"""
    # Preprocess image
    # image = PIL.Image.resize(image, (28, 28))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    print(np.max(image))
    image = image.astype(np.float32) / 255.
    print(np.max(image))
    input_image = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
    print(input_image.shape)
    # result = cnn_net.infer(inputs={input_layer: input_image})
    # result_ir = result[output_layer]
    result_ir = ort_session.run(None, {'input': input_image})[0]
    result = np.argmax(result_ir, axis=1)
    # Prepare data for visualization
    prediction = np.argmax(result_ir, axis=1)[0]
    print(result, result_ir, prediction)
    return result



if __name__ == '__main__':
    # ie  = IECore()

    args = parse_args()

    # Load the ONNX model
    model = onnx.load(args.model)
    # Check that the model is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))
    # Load the ONNX model
    ort_session = ort.InferenceSession(args.model)

    # cnn = ie.read_network(model=args.model)
    # cnn_net = ie.load_network(network=cnn, device_name=args.device)
    # input_layer = next(iter(cnn_net.input_info))
    # output_layer = next(iter(cnn_net.outputs))
    # # print(input_layer, output_layer)

    title = "CNN MNIST Classifier" 
    description = "Classify MNIST digits using CNN"
    iface = gr.Interface(
        cnn_classifier,
        [
            gr.components.Image(
                shape=None,
                image_mode="L",  
                invert_colors=False,
                source="upload",
                tool="editor",
                type="filepath",
                label='MNIST Image')],
        [
            gr.components.Textbox(type="auto", label='Prediction'),
        ],
        title=title,
        description=description,
        )

    iface.launch(server_name="0.0.0.0", server_port=int(os.getenv('PORT', "8150")))
