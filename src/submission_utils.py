import os
import inspect
from datetime import datetime

def save_submission(df, run_name=None):
    """
    Saves submission into the ROOT submissions/ folder.
    If run_name is given, it is used in the filename.
    """

    # Figure out the base name
    if run_name is None:
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_name = os.path.splitext(os.path.basename(caller_file))[0]
    else:
        caller_name = run_name  # e.g. "felipe_model"

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    filename = f"{caller_name}__{timestamp}.csv"

    # save to ROOT/submissions/
    root_submissions = os.path.abspath(os.path.join(
        os.path.dirname(__file__),# src/
        "..",  # project root
        "submissions"
    ))
    os.makedirs(root_submissions, exist_ok=True)

    full_path = os.path.join(root_submissions, filename)
    df.to_csv(full_path, index=False)

    print(f"Saved submission -> {full_path}")