import json
import logging
from typing import Union

from flask import Flask, jsonify, request, Response

from preprocessor import PropagationPreprocessor

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/process_dataset", methods=['POST'])
    def process_dataset() -> Union[str, Response]:

        dbname = request.args.get('db_name')
        if not dbname:
            return jsonify({"error": "No database name was provided"}), 400


        data = request.files['file']

        if not data:
            return jsonify({"error": "No JSON object was provided"})

        try:
            dataset_data = data.read().decode('utf-8')
            dataset_data = [json.loads(line) for line in dataset_data.strip().split('\n')]
        except KeyError:
            logger.error("No dataset data was provided")
            return jsonify({"error": "No dataset data was provided"})
        except Exception as e:
            logger.error(f"Error: {e}")
            return jsonify({"error": f"Error: {e}"})

        try:
            propagation_preprocessor = PropagationPreprocessor(dataset=dbname, data=dataset_data, host="mongo")
            propagation_preprocessor.process()
            logger.info(f"Processed dataset {dbname}")
            return jsonify({"message": f"Processed dataset {dbname}"})
        except Exception as e:
            logger.error(f"Error: {e}")
            return jsonify({"error": f"Error: {e}"})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
