from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Demo FastAPI")

class Item(BaseModel):
    id: int
    name: str
    desc: Optional[str] = None
    price: str

items_db = [
    {
        "id": 1,
        "name": "Coca",
        "desc": "Đồ uống có ga",
        "price": "15.000 VND"
    }, 
    {
        "id": 2,
        "name": "Pepsi",
        "desc": "Đồ uống có ga", 
        "price": "14.000 VND"
    }, 
]


@app.get("/demo")
def read_root():
    return {"message": "Welcome to FastAPI"}


@app.get("/items/", response_model=List[Item])
def read_items():
    return items_db


@app.get("/items/{item_id}", response_model=Item)
def read_item(item_id: int):
    item = next((item for item in items_db if item['id'] == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@app.post("/items/", response_model=Item, status_code=201)
def create_item(item: Item):
    if any(x['id'] == item.id for x in items_db): # existed
        raise HTTPException(status_code=400, detail="Item ID already exists")
    items_db.append(item)
    return item # optional


@app.put("/items/{item_id}", response_model=Item, status_code=202)
def update_item(item_id: int, item: Item):
    index = next((i for i, item in enumerate(items_db) if item['id'] == item_id), None)
    if index is not None:
        items_db[index] = item
        return item
    raise HTTPException(status_code=404, detail="Item not found")


@app.delete("/items/{item_id}", status_code=203)
def delete_item(item_id: int):
    index = next((i for i, item in enumerate(items_db) if item['id'] == item_id), None)
    if index is None:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db.pop(index)
    return {"message": "Item deleted"}