import base64
import json

import requests
from dash import dcc, html, Output, Input

from components.components import RemissComponent
import dash_bootstrap_components as dbc


class UploadComponent(RemissComponent):
    def __init__(self, target_api_url, name=None):
        super().__init__(name)
        self.target_api_url = target_api_url

        self.upload = dcc.Upload(
            id=f'upload-{self.name}',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select jsonl File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
        )
        self.feedback = html.Div(id=f'feedback-{self.name}')

    def layout(self, params=None):
        return self.upload

    def process_upload(self, contents, filename):
        if contents is None:
            return dbc.Alert('No file uploaded', color='warning')
        if filename.endswith('.jsonl'):
            try:
                data = self.decode_jsonl(contents)
                self.send_to_api(data)
                return dbc.Alert('File uploaded', color='success')
            except Exception as e:
                return dbc.Alert(f'Error uploading file: {e}', color='danger')
        else:
            return dbc.Alert('Invalid jsonl file format.', color='danger')

    def decode_jsonl(self, contents):
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        return [json.loads(line) for line in decoded.decode('utf-8').split('\n') if line]

    def send_to_api(self, data):
        requests.post(self.target_api_url, json=data)

    def callbacks(self, app):
        app.callback(
            Output(f'feedback-{self.name}', 'children'),
            Input(f'upload-{self.name}', 'contents'),
            prevent_initial_call=True
        )(self.process_upload)