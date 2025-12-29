import argparse

from datasets import load_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates",
        type=str,
        default="cmrc2018,dureader,DuReader,DuReader_robust,du_reader_robust,dureader_robust,DRCD,webqa",
        help="Comma-separated HF dataset ids to probe",
    )
    args = p.parse_args()

    candidates = [x.strip() for x in args.candidates.split(",") if x.strip()]
    for name in candidates:
        try:
            ds = load_dataset(name)
            print("OK", name, "splits", list(ds.keys()))
        except Exception as e:
            msg = str(e).replace("\n", " ")
            print("NO", name, type(e).__name__, msg[:200])


if __name__ == "__main__":
    main()


