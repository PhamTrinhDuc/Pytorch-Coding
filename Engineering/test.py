import requests


API = "http://127.0.0.1:8000"


def GET_methods(path: str, id: int = None):

    if id is not None:
        response = requests.get(url=f"{API}/{path}/{id}")
    else:
        response = requests.get(url=f"{API}/{path}")
        
    if response.status_code == 200:
        print(f"GET data by path: [{path}] successfull: \n {response.json()}")
    else:
        print(f"request path; [{path}] failed. Status code: {response.status_code}: ")
        print(response.content)


def POST_methods(path: str, payload: dict):
    response = requests.post(url=f"{API}/{path}", json=payload)
    if response.status_code == 203:
        print(f"POST data by path: [{path}] successfull. Status code: {response.status_code}")
        print(response.json())
    else:
        print(f"request path: [{path}] failed. Status code: {response.status_code}: ")
        print(response.content)


def PUT_method(path: str, id: int, payload: dict):
    response = requests.put(url=f"{API}/{path}/{id}", json=payload)
    if response.status_code == 202:
        print(f"PUT data by path: [{path}] successfull. Status code: {response.status_code}")
        print(response.json())
    else:
        print(f"request path: [{path}] failed. Status code: {response.status_code}: ")
        print(response.content)


def DELTE_method(path: str, item_id: int):
    response = requests.delete(url=f"{API}/{path}/{item_id}")
    if response.status_code == 203:
        print(f"DELETE data by path: [{path}] successfull. Status code: {response.status_code}")
        print(response.json())
    else:
        print(f"request path: [{path}] failed. Status code: {response.status_code}: ")
        print(response.content)


if __name__ == "__main__":

    # =================== test GET method
    # gets = ["demo", "items", ("items", 1)]
    # GET_methods(*gets[2])

    # =================== test GET method
    # posts = ["items"]
    # payload = {
    #     "id": 3, 
    #     "name": "Gà rán", 
    #     "desc": "Đồ ăn nhanh chiên rán", 
    #     "price": "56.000 VND"
    # }
    # POST_methods(path=posts[0], payload=payload)

    # ==================== test PUT method
    # payload = {
    #     "id": 1,
    #     "name": "Coca",
    #     "desc": "Đồ uống có ga",
    #     "price": "18.000 VND"
    # }
    # PUT_method(path="items", id=1, payload=payload)
    # ==================== test DELETE method
    DELTE_method(path="items", item_id=2)