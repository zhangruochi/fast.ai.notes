from starlette.applications import Starlette
from starlette.responses import JSONResponse
# from starlette.routing import Route
import uvicorn
import torch
from pathlib import Path
from io import BytesIO
import sys
import aiohttp
import asyncio
import fastai



from fastai.vision import *
defaults.device = torch.device('cpu')
learner = load_learner(".")



app = Starlette(debug=False)

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()



@app.route("/", methods=["GET"])
async def homepage(request):
    return JSONResponse({'hello': 'world'})


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,outputs = learner.predict(img)
    return JSONResponse({
        "predictions":  str(pred_class)})




if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)

