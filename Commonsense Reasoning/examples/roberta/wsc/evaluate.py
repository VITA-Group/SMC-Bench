from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
import os, re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


check_point_folder = '/home/sliu/project_space/pruning_fails/QA/robert/winogrande/WSC/'
data_dir = '/home/sliu/project_space/pruning_fails/QA/robert/winogrande/winogrande_1.1/'

for method in ['random/']:

    check_point_folder = check_point_folder + method
    # model_files = [0.5]
    model_files = os.listdir(check_point_folder)
    model_files = sorted_nicely(model_files)

    for file in model_files:
        print(file)
        roberta = RobertaModel.from_pretrained(check_point_folder+str(file), 'checkpoint_best.pt', data_dir)
        roberta.cuda()
        nsamples, ncorrect = 0, 0
        for sentence, pronoun_span, query, cand in wsc_utils.winogrande_jsonl_iterator(data_dir+'val.jsonl', eval=True):
            pred = roberta.disambiguate_pronoun(sentence)
            print(pred)
            # nsamples += 1
            # if pred == label:
            #     ncorrect += 1
        print('Accuracy: ' + str(ncorrect / float(nsamples)))