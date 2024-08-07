import logging
from typing import Union

from flask import Flask, jsonify, request, Response
from pymongo import MongoClient

from preprocessor import PropagationPreprocessor

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route("/process_dataset", methods=['POST'])
def process_dataset() -> Union[str, Response]:
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON object was provided"})

    # Store data in mongodb raw
    try:
        dataset_name = data.get("dataset_name")
    except KeyError:
        logger.error("No dataset name was provided")
        return jsonify({"error": "No dataset name was provided"})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": f"Error: {e}"})

    try:
        dataset_data = data.get("dataset_data")
    except KeyError:
        logger.error("No dataset data was provided")
        return jsonify({"error": "No dataset data was provided"})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": f"Error: {e}"})

    try:
        propagation_preprocessor = PropagationPreprocessor(dataset=dataset_name, data=dataset_data)
        propagation_preprocessor.process()
        logger.info(f"Processed dataset {dataset_name}")
        return jsonify({"message": f"Processed dataset {dataset_name}"})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": f"Error: {e}"})
