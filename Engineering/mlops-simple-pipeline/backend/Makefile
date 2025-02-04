mlflow_up:
	docker compose -f ./docker/mlflow/docker-compose.yml up -d 

mlflow_down:
	docker compose -f ./docker/mlflow/docker-compose.yml down 

serving_up:
	MODEL_NAME=${model_name} MODEL_ALIAS=${model_alias} PORT=${port} DEVICE=cpu docker compose -f ./docker/model_serving/docker-compose.yml up -d

serving_down: 
	MODEL_NAME="" MODEL_ALIAS="" PORT=8000 DEVICE="" docker compose -f ./docker/model_serving/docker-compose.yml down
