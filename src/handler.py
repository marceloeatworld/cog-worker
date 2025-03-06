import time
import subprocess
import traceback

import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
from runpod import RunPodLogger

# Initialize logger
log = RunPodLogger()

LOCAL_URL = "http://127.0.0.1:5000"

# Configure session with retries for transient errors
cog_session = requests.Session()
retries = Retry(
    total=10,
    backoff_factor=0.1,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET", "POST"]
)
cog_session.mount('http://', HTTPAdapter(max_retries=retries))

# ----------------------------- Start API Service ---------------------------- #
# Call "python -m cog.server.http" in a subprocess to start the API service.
subprocess.Popen(["python", "-m", "cog.server.http"])


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url, max_retries=None):
    '''
    Check if the service is ready to receive requests.
    If max_retries is None, wait indefinitely.
    '''
    retries = 0
    while max_retries is None or retries < max_retries:
        try:
            health = requests.get(url, timeout=120)
            status = health.json()["status"]

            if status == "READY":
                log.info("COG service is ready")
                time.sleep(1)
                return True

        except requests.exceptions.RequestException:
            if max_retries:
                log.info(f"Service not ready yet. Retrying... ({retries+1}/{max_retries})")
            else:
                log.info("Service not ready yet. Retrying...")
        except Exception as err:
            log.warn(f"Health check error: {str(err)}")

        retries += 1
        time.sleep(0.5)
    
    log.error(f"Service failed to start after {max_retries} retries")
    return False


def run_inference(inference_request):
    '''
    Run inference on a request with improved error handling.
    '''
    try:
        # Use the same timeout as before for consistency
        response = cog_session.post(
            url=f'{LOCAL_URL}/predictions',
            json=inference_request, 
            timeout=600  # 10 minutes timeout
        )
        
        # Check status code before attempting to parse JSON
        if response.status_code == 200:
            return response.json()
        else:
            log.error(f"Error response from COG: {response.status_code}")
            error_message = response.text[:200] if response.text else "Unknown error"
            
            # Try to extract error details from JSON if possible
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_message = error_json["error"]
            except:
                pass
                
            # Return a properly structured error response
            return {"output": {"error": error_message}}
            
    except requests.exceptions.Timeout:
        log.error("Request timed out after 600 seconds")
        return {"output": {"error": "Generation timed out after 10 minutes"}}
        
    except requests.exceptions.RequestException as e:
        log.error(f"Request error: {str(e)}")
        return {"output": {"error": f"Request failed: {str(e)}"}}
        
    except Exception as e:
        log.error(f"Unexpected error: {str(e)}")
        log.debug(traceback.format_exc())
        return {"output": {"error": f"Unexpected error: {str(e)}"}}


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    try:
        # Extract input from event
        job_input = event.get("input", {})
        log.info("Received new job request")
        
        # Optional: Add some validation for the input
        if not job_input:
            log.warn("Empty input received")
            return {"error": "No input provided"}

        # Run inference
        result = run_inference({"input": job_input})
        
        # Return the output as before
        if "output" in result:
            log.info("Job completed successfully")
            return result["output"]
        else:
            log.error("No output in response")
            return {"error": "No output in response"}
            
    except Exception as e:
        log.error(f"Handler error: {str(e)}")
        log.debug(traceback.format_exc())
        return {"error": f"Handler error: {str(e)}"}


if __name__ == "__main__":
    log.info("Starting COG API service")
    
    # Wait for service to be ready (without a retry limit)
    log.info("Waiting for COG service to be ready...")
    wait_for_service(url=f'{LOCAL_URL}/health-check')
    
    log.info("COG API Service is ready. Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})