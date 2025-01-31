# inspiration: 
# https://github.com/aarushdixit889/photo-semantics-analyzer/blob/main/app.py
# https://medium.com/@codingmatheus/sending-images-to-claude-3-using-amazon-bedrock-b588f104424f
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps
# https://github.com/awsdocs/aws-doc-sdk-examples/blob/main/python/example_code/bedrock-runtime/models/anthropic_claude/converse.py#L4
# https://docs.aws.amazon.com/nova/latest/userguide/modalities-image-examples.html
# https://discuss.streamlit.io/t/unique-key-for-every-items-in-radio-button/20654/4
# https://stackoverflow.com/questions/3715493/encoding-an-image-file-with-base64
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html
# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Message.html
# https://docs.anthropic.com/en/docs/build-with-claude/vision


import streamlit as st
import json
from PIL import Image
from PIL import ImageOps
import PIL
import os
import base64
import boto3
from botocore.exceptions import ClientError


# -- functions --
def call_llm(model_id, ref_image, up_image_bytes, language):
    
    system_msgs = [
        {
        "text":
        "You are an expert medical doctor. When the user provides you with an image of their urine test strip, analyze carefully the color of the various indicators on the test and compare it to the testkit reference. Then provide a short medical analysis and lookout for possible infection indicators. Provide your answer in a concise format. Provide your answer in markdown format. Do not analyze images that are not containing a urine test strip. Always end the response with a disclaimer that this is not a medical advice. Please respond in the following language: " + language 
        }
    ]
    
    #inferenceParams = {}
    msg_step1 = "Create a table containing the list of parameters (top to bottom) from this uring test reference, and their color indicator for a normal state:"
    
    msg_step2 = "Step by step, create a list of colors you detect in this second image of a used test strip (top to bottom) and compare it to the reference image. Add the result to the previous table."
    
   
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"text": msg_step1},
                {"image":{"format":"jpeg", "source":{"bytes": ref_image}}},
                {"text": msg_step2},
                {"image":{"format":"jpeg", "source":{"bytes": up_image_bytes}}},
            ],
            }
        ]
    
    try:
        # Send the message to the model, using a basic inference configuration.
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            system=system_msgs,
            #inference...
        )
    
        # Extract and print the response text.
        response_text = response["output"]["message"]["content"][0]["text"]
        st.write(response_text)
    
    except (ClientError, Exception) as e:
        st.write(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    return

#--- end of function definition ---


####--- main page ---###
st.title(":pill: Urine Test Analysis v2")
st.write("Upload a photo of your urine test strip for analysis")

#--Select model for inference
# naming conventions: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
model_ids = ["us.anthropic.claude-3-5-sonnet-20240620-v1:0", "amazon.nova-lite-v1:0"]
model_id = model_ids[0] 
st.write("\(note: this app uses the following LLM model: ", model_id, "\)" )

#--Create a Bedrock Runtime client in the AWS Region you want to use.
client = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_REGION"]
)

#--select user language
output_language = st.radio("Select your language:", ["Spanish","Italian","English"]) 
st.write("You selected: ", output_language)


#--display and open test reference image
st.sidebar.image("uri_test_reference.jpeg")

#--open ref image as bytes for LLM model
with open("uri_test_reference.jpeg", "rb") as f:
        ref_image = f.read()

#--upload test strip photo, rotate it, save it
up_image=st.file_uploader("Upload your photo oriented as the reference", type=["jpeg", "png"])
if up_image is not None:
    st.image(up_image)
    #-read uploaded image as bytes for llm model
    up_image_bytes = up_image.read()
    call_llm(model_id, ref_image, up_image_bytes, output_language)


