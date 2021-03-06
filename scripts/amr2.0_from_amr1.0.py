import sys
import argparse
from collections import defaultdict, Counter
from numpy.random import choice
import numpy as np


def argument_parser():

    parser = argparse.ArgumentParser(description='AMR parser oracle')
    # Single input parameters
    parser.add_argument(
        "--in-amr",
        help="AMR 2.0+ annotation file to be splitted",
        type=str,
        required=True
    )
    parser.add_argument(
        "--in-amr1",
        help="AMR 1.0 annotations file to be used for reference",
        type=str,
        required=True
    )
    parser.add_argument(
        "--out-amr-from-amr1",
        help="Split with a subset of AMR1.0 ids",
        type=str,
    )
    parser.add_argument(
        "--out-amr-reminder",
        help="A subset of the --in-amr not in --out-amr-from-amr1",
        type=str,
    )
    parser.add_argument(
        "--max-sentences-amr",
        help="Maximum of sentences for amr-labeled set. Subset equaaly "
        "sampled from amr1 domains",
        type=int,
    )
    args = parser.parse_args()

    return args


def read_amr_as_raw(file_path):
    with open(file_path) as fid:
        raw_amrs = []
        raw_amr = []
        for line in fid.readlines():
            if line.strip():
                raw_amr.append(line.strip()) 
            else:
                raw_amrs.append(raw_amr)
                raw_amr = []
    return raw_amrs


def write_amr_from_raw(out_file, raw_amrs):
    with open(out_file, 'w') as fid:
        for raw_amr in raw_amrs:
            fid.write('\n'.join(raw_amr))
            fid.write('\n\n')


if __name__ == '__main__':

    # Argument handling
    args = argument_parser()

    # read data
    # For AMr1.0 see https://catalog.ldc.upenn.edu/docs/LDC2014T12/README.txt
    in_amr2 = read_amr_as_raw(args.in_amr)
    in_amr1 = read_amr_as_raw(args.in_amr1)

    # store by id
    amr2_by_id = {raw_amr[0].split()[2]: raw_amr for raw_amr in in_amr2}
    amr1_by_id = {raw_amr[0].split()[2]: raw_amr for raw_amr in in_amr1}

    # additional by id and sub-domain
    ids_by_domain = defaultdict(list)
    for corpus_id, _ in amr1_by_id.items():
        domain = corpus_id.split('_')[0].split('-')[0]
        ids_by_domain[domain].append(corpus_id)

    # Use all AMR1.0 ids or a subset but with sample balanced across domains
    if args.max_sentences_amr and len(amr1_by_id) > args.max_sentences_amr:
        domains = list(ids_by_domain.keys())
        ratio = args.max_sentences_amr / len(amr1_by_id)
        labeled_ids = []
        for domain in domains[:-1]:
            max_sent = int(np.floor(len(ids_by_domain[domain])*ratio))
            new_ids = list(choice(
                ids_by_domain[domain], max_sent, replace=False
            ))
            labeled_ids.extend(new_ids)
        remanining_len = args.max_sentences_amr - len(labeled_ids) - 1
        new_ids = list(choice(
            ids_by_domain[domains[-1]], remanining_len, replace=False
        ))
        labeled_ids.extend(new_ids)
    else:
        labeled_ids = list(amr1_by_id.keys())

    # split AMR 2.0
    amr2_from_amr1 = []
    amr2_minus_amr1 = []
    for index, raw_amr in enumerate(in_amr2):
        sentence_id = raw_amr[0].split()[2]
        if sentence_id in labeled_ids:
            amr2_from_amr1.append(raw_amr)
        else:
            amr2_minus_amr1.append(raw_amr)

    # write
    print(f'Writing {args.out_amr_from_amr1}')
    write_amr_from_raw(args.out_amr_from_amr1, amr2_from_amr1)
    print(f'Writing {args.out_amr_reminder}')
    write_amr_from_raw(args.out_amr_reminder, amr2_minus_amr1)
