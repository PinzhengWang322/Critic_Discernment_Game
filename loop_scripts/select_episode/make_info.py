import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--loop', type=int)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        old_data_info = {}
    else:
        old_data_info = json.load(open(args.path))

    new_data_info = {
        f"correct-round2-loop{args.loop}": {
            "file_name": f"correct-prover-round2-loop{args.loop}.json",
            "formatting": "sharegpt",
            "columns": {
            "messages": "conversations",
            }
        },
        f"mislead-round2-loop{args.loop}": {
            "file_name": f"mislead-prover-round2-loop{args.loop}.json",
            "formatting": "sharegpt",
            "columns": {
            "messages": "conversations",
            }
        },
        f"EI_round1-loop{args.loop}": {
            "file_name": f"EI-loop{args.loop}.json",
            "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
          }
        },
        f"mislead_critic-loop{args.loop}": {
            "file_name": f"mislead-critic-loop{args.loop}.json",
            "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
          }
        },
        f"correct_critic-loop{args.loop}": {
            "file_name": f"correct-critic-loop{args.loop}.json",
            "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
          }
        }
    }

    old_data_info.update(new_data_info)
    with open(args.path, 'w') as f:
        f.write(json.dumps(old_data_info, indent=4))