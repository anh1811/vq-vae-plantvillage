import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
import zipfile
import shutil
from model_handler import ModelHandler

app = FastAPI()
MODEL_TYPE = os.getenv("MODEL_TYPE", "default")

@app.post("/fine-tune-and-generate")
async def main(
    dataset: UploadFile = File(...),
    model_type: str = Form(...),
    epochs: int = Form(10),
    batch_size: int = Form(32),
):
    if model_type != MODEL_TYPE:
        raise HTTPException(400, f"Model type {model_type} not supported by this container")

    # Process dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save uploaded file
        dataset_path = f"{tmp_dir}/dataset.zip"
        with open(dataset_path, "wb") as f:
            f.write(await dataset.read())
        
        # Extract dataset
        extracted_dir = f"{tmp_dir}/extracted"
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(extracted_dir)
        
        # Train model
        handler = ModelHandler()
        handler.fine_tune(extracted_dir, epochs, batch_size)
        
        # Generate data
        synthetic_data = handler.generate()
        
        # Create output zip
        output_dir = f"{tmp_dir}/output"
        os.makedirs(output_dir, exist_ok=True)
        synthetic_data.to_csv(f"{output_dir}/synthetic_data.csv", index=False)
        
        # Zip results
        output_zip = f"{tmp_dir}/syntheticdata.zip"
        with zipfile.ZipFile(output_zip, "w") as zipf:
            zipf.write(f"{output_dir}/synthetic_data.csv", arcname="synthetic_data.csv")

        return FileResponse(output_zip, filename="syntheticdata.zip")