from waitress import serve
import B2113307

if __name__ == "__main__":
    serve(B2113307.app, host='0.0.0.0', port=8000)
