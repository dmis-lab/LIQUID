import argparse
import hashlib
import json
import os
import spacy
import requests
import transformers

from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

from datetime import datetime
from collections import defaultdict
from tqdm import tqdm


transformers.logging.set_verbosity_error()

def summarization(data, model, batch_size, min_length, max_length):
    summary = []
    context = []
    titles = []
    num_paragraphs = []
    summary_context = {
        "data": []
    }

    for i in range(len(data)):
        title = data[i]['title']
        paragraphs = data[i]['paragraphs']
        titles.append(title)
        num_paragraphs.append(len(paragraphs))

        for text in paragraphs:
            text = text['context'].strip()
            context.append(text[:768])

    print("********** Context Summarization starts! **********")
    
    for i in tqdm(range(0, len(context), batch_size), total=len(context)//batch_size):
        if i+batch_size <= len(context):
            batch = context[i:i+batch_size]
        else:
            batch = context[i:]
        result = model(batch, min_length=0, max_length=max_length, batch_size=batch_size)
        summary += result

    print("********** Context Summarization ends! **********")

    summary = [x['summary_text'] for x in summary]
    assert len(summary) == len(context), "Summarization is inconsistency!"

    start_idx = 0
    for i in range(len(titles)):
        title = titles[i]
        num_context = num_paragraphs[i]
        summaries = summary[start_idx:start_idx+num_context]
        original_text = context[start_idx:start_idx+num_context]
        paragraphs = [{"context": s, "original_text": o} for s, o in zip(summaries, original_text)]
        summary_context['data'].append({
            "title": title,
            "paragraphs": paragraphs
        })
        start_idx += num_context

    return summary_context


def bern2(text, url="http://localhost:8888/plain"):
    return requests.post(url, json={'text': text}).json()

def answer_extraction(is_biomedical, data, do_summary):
    print("********** Answer extraction starts! **********")
    temp = []
    for d in data:
        title = d['title']
        paragraphs = d['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            if do_summary:
                original_text = paragraph['original_text']
                temp.append({
                    "title": title,
                    "context": context,
                    "original_text": original_text
                })
            else:
                temp.append({
                    "title": title,
                    "context": context,
                })
    
    if not is_biomedical:
        model = spacy.load('en_core_web_sm')
    
    ca = {
        "data": []
    }

    for i in tqdm(range(len(temp)), total=len(temp)):
        context = temp[i]['context']
        original_text = temp[i]["original_text"] if do_summary else temp[i]['context'] 

        title = temp[i]['title']
        
        if is_biomedical:
            ents = bern2(context)['annotations']
        else:
            ents = model(context)
            ents = ents.ents
        
        ents_set = set()
        type2ent = defaultdict(set)
        
        for ent in ents:
            if is_biomedical:
                label = ent['obj']
                ent = str(ent['mention'])
                if label != 'species':
                    type2ent[label].add(ent)
                    ents_set.add((ent, label))

            else:
                label = str(ent.label_)
                if label != 'DATE':
                    ent = str(ent)
                    type2ent[label].add(ent)
                    ents_set.add((ent, label))

        for ent_type, entity in type2ent.items():
            entity = list(entity)
            if 1 < len(entity):
                answers = []
                flag = True
                
                for ent in entity:
                    answer_start = original_text.find(ent)
                    if answer_start == -1:
                        flag = False
                    answer = {
                        "answer_text": ent,
                        "answer_type": ent_type,
                        "answer_start": answer_start
                    }
                    if flag:
                        answers.append(answer)
                if 1 < len(answers):
                    cas = {
                        "title": title,
                        "context": original_text,
                        "answers": answers,
                    } 

                    ca['data'].append(cas)
    
    print("********** Answer extraction ends! **********")
    
    return ca


def question_generation(model, data, batch_size, min_length, max_length):
    
    print("********** Question Generation Starts! **********")
    ca = []
    for i in range(len(data)):
        answers = [x['answer_text'] for x in data[i]['answers']]
        answers_string = ", ".join(answers)
        context = data[i]['context']
        text = "answers: %s context: %s </s>" % (answers_string, context)
        ca.append(text)
    
    questions = []
    for i in tqdm(range(0, len(ca), batch_size), total=len(ca)//batch_size):
        if i+batch_size <= len(ca):
            result = model(ca[i:i+batch_size], batch_size=batch_size)
        else:
            result = model(ca[i:], batch_size=batch_size)
        questions += result
    
    # {'generated_text': 'question: What is AI?'}
    q_start_idx = 10
    questions = [q['generated_text'][q_start_idx:] for q in questions]
    assert len(questions) == len(data), "Question Generation is inconsistency!"

    cqa = {
        "data": []
    }
    
    title2ctx = {}
    ctx2qas = defaultdict(list)
    titles = [x['title'] for x in data]

    for title in titles:
        title2ctx[title] = []

    for i in range(len(data)):
        answers = data[i]['answers']
        question = questions[i]
        context = data[i]['context']
        title = data[i]['title']
        
        str2hash = title + context + question + "".join([a["answer_text"] for a in answers])
        hash_res = hashlib.md5(str2hash.encode())
        qid = hash_res.hexdigest()

        title2ctx[title].append(context)
        ctx2qas[context].append({
            "id": qid,
            "question": question,
            "answers": answers
        })
    
    for title, ctx in title2ctx.items():
        temp = {
            "title": title,
            "paragraphs": []
        }
        for c in ctx:
            qas = ctx2qas[c]
            paragraph = {
                "context": c,
                "qas": qas
            }
            if paragraph not in temp['paragraphs']:
                temp["paragraphs"].append(paragraph)
            
        cqa['data'].append(temp)

    print("********** Question Generation Ends! **********")

    return cqa

def answer_filtering(answers, pseudo_answers, context, filter_count):

    # filtering
    filtered_answers = []
    answer_start = []
    min_prob = 10
    for answer in answers:
        for pseudo_answer in pseudo_answers:
            try:
                if 0.01 <= pseudo_answer['score']:
                    if answer in pseudo_answer['answer']:
                        min_prob = min(pseudo_answer['score'], min_prob)
                        filtered_answers.append(answer)
                        answer_start.append(pseudo_answer['start'])
                        break
                else:
                    if answer == pseudo_answer['answer']:
                        min_prob = min(pseudo_answer['score'], min_prob)
                        filtered_answers.append(answer)
                        answer_start.append(pseudo_answer['start'])
                        break 
            except:
                pass
        
    if filter_count != 2:
        assert len(answer_start) == len(filtered_answers), f"len answer_text ({len(filtered_answers)}) != len answer_start ({len(answer_start)})"
        
        if len(filtered_answers) == 0:
            context = ""

        return filtered_answers, answer_start, context

    # expansion
    else:
        if 1 < len(filtered_answers):
            for pseudo_answer in pseudo_answers:
                if pseudo_answer['score'] < min_prob:
                    break
                try:
                    is_in = False
                    for filtered_answer in filtered_answers:
                        if filtered_answer in pseudo_answer['answer']:
                            is_in = True
                            break
                    if is_in is False and context.find(pseudo_answer['answer']) != -1:
                        filtered_answers.append(pseudo_answer['answer'])
                        answer_start.append(pseudo_answer['start'])
                except:
                    pass
        
        # deduplication
        final_filtered_answers = []
        final_answer_start = []
        for i in range(len(filtered_answers)):
            flag = False
            for j in range(len(filtered_answers)):
                if i != j:
                    filtered_answers_i = filtered_answers[i].lower()
                    filtered_answers_j = filtered_answers[j].lower()
                    if filtered_answers_i in filtered_answers_j:
                        flag = True
                        break

            if flag is False:
                final_filtered_answers.append(filtered_answers[i])
                final_answer_start.append(answer_start[i])


        assert len(final_answer_start) == len(final_filtered_answers), f"len answer_text ({len(final_filtered_answers)}) != len answer_start ({len(final_answer_start)})"
        
        if len(final_filtered_answers) == 0:
            context = ""

        return final_filtered_answers, final_answer_start, context

def iterative_filtering(data, qg_model, qa_model, batch_size):

    print("********** Iterative Filtering Starts! **********")

    q_start_idx = 10
    data_list = []

    for i in range(len(data)):
        title = data[i]['title']
        paragraphs = data[i]['paragraphs']
        for paragraph in paragraphs:
            context = paragraph['context']
            qas = paragraph['qas']
            for qa in qas:
                qid = qa['id']
                question = qa['question']
                answers = qa['answers']
                data_list.append({
                    "id": qid,
                    "title": title,
                    "context": context,
                    "question": question,
                    "answers": answers
                })

    title2ctx = defaultdict(set)
    ctx2qas = defaultdict(list)

    for i in tqdm(range(0, len(data_list), batch_size), total=len(data_list)//batch_size):
        batch = data_list[i:i+batch_size] if i+batch_size <= len(data_list) else data_list[i:]
        context = [x['context'] for x in batch]
        question = [x['question'] for x in batch]
        answers = [x['answers'] for x in batch]
        qid = [x['id'] for x in batch]
        title = [x['title'] for x in batch]

        answer_text = []
        for answer in answers:
            answer_text.append([x['answer_text'] for x in answer])

        generated_q = []
        for idx in range(3):
            answer_string = [", ".join(x) for x in answer_text]
            ca = ["answers: %s context: %s </s>" % (a, c) for a, c in zip(answer_string, context)]
            generated_q = qg_model(ca, batch_size=batch_size)
            generated_q = [x['generated_text'][q_start_idx:] for x in generated_q]
            pseudo_answers = qa_model(question=generated_q, context=context, top_k=30, batch_size=batch_size)
            filtered_answer_text = []
            answer_starts = []
            filtered_contexts = []
            filtered_titles = []
            filtered_qids = []

            for answer, ctx, q, t, pseudo_answer in zip(answer_text, context, qid, title, pseudo_answers):
                filtered_answers, answer_start, filtered_context = answer_filtering(answer, pseudo_answer, ctx, idx)
                if filtered_context != "":
                    filtered_contexts.append(filtered_context)
                    filtered_answer_text.append(filtered_answers)
                    answer_starts.append(answer_start)
                    filtered_qids.append(q)
                    filtered_titles.append(t)

            
            answer_text = filtered_answer_text
            context = filtered_contexts
            qid = filtered_qids
            title = filtered_titles
        

        assert len(answer_starts) == len(answer_text) == len(context) == len(qid) == len(title), \
            f"len answer_text ({len(answer_text)}) != len answer_start ({len(answer_starts)}) != \
              len context ({len(context)}) != len qid ({len(qid)}) != len({len(title)})"
        
        
        # final question generation
        answer_string = [", ".join(x) for x in answer_text]
        ca = ["answers: %s context: %s </s>" % (a, c) for a, c in zip(answer_string, context)]
        new_generated_q = qg_model(ca, batch_size=batch_size)
        new_generated_q = [x['generated_text'][q_start_idx:] for x in new_generated_q]
        pseudo_answers = qa_model(question=new_generated_q, context=context, top_k=30, batch_size=batch_size)
        
        # final filtering
        filtered_answer_text = []
        answer_starts = []
        filtered_contexts = []
        filtered_titles = []
        filtered_qids = []
        final_questions = []

        for answer, ctx, q, t, pseudo_answer, oq, nq in zip(answer_text, context, qid, title, pseudo_answers, generated_q, new_generated_q):
            filtered_answers, answer_start, filtered_context = answer_filtering(answer, pseudo_answer, ctx, 0)
            if filtered_context != "":
                if set(filtered_answers) == set(answer):
                    final_questions.append(nq)
                else:
                    final_questions.append(oq)

                filtered_contexts.append(filtered_context)
                filtered_answer_text.append(filtered_answers)
                answer_starts.append(answer_start)
                filtered_qids.append(q)
                filtered_titles.append(t)
        
        assert len(final_questions) == len(answer_starts) == len(filtered_answer_text) == len(filtered_contexts), \
            f"len final_questions {len(final_questions)} != len answer_starts {len(answer_starts)} \
                != len answer_text {len(filtered_answer_text)} != len contexts {len(filtered_contexts)}"

        qid = filtered_qids
        title = filtered_titles
        context = filtered_contexts
        generated_q = final_questions
        answer_text = filtered_answer_text

        for j in range(len(answer_text)):
            filtered_answers = answer_text[j]
            if 1 < len(filtered_answers):
                answers = [{"answer_text": ans, "answer_start": ans_start} for ans, ans_start in zip(filtered_answers, answer_starts[j])]
                title2ctx[title[j]].add(context[j])
                ctx2qas[context[j]].append({
                    "id": qid[j],
                    "question": generated_q[j],
                    "answers": answers
                })
    
    filtered_data = {
        "data": []
    }

    for title, ctxs in title2ctx.items():
        tp = {
            "title": title,
            "paragraphs": []
        }
        for ctx in ctxs:
            cqa = {
                "context": ctx,
                "qas": ctx2qas[ctx]
            }
            tp['paragraphs'].append(cqa)
        filtered_data['data'].append(tp)
    
    print("********** Iterative Filtering Ends! **********")

    return filtered_data

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_data(file_path, data):
    # extracdt path
    dir_name = os.path.dirname(file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Saved the dataset at '{}'.".format(file_path))

def main(args):
    # count
    data = load_data(args.data_file)

    if args.doc_limit:
        data = {
            "data": data["data"][:args.doc_limit]
        }

    # summarization
    if args.do_summary:
        sum_tokenizer = BartTokenizer.from_pretrained(args.summary_model_name_or_path)
        sum_model = BartForConditionalGeneration.from_pretrained(args.summary_model_name_or_path)
        sum_pipe = pipeline('summarization', model=sum_model, tokenizer=sum_tokenizer, device=args.device, truncation=True)

        context = summarization(data['data'], sum_pipe, args.batch_size, args.summary_min_length, args.summary_max_length)
        #now = datetime.now()
        #with open(f'{output_file}/context-summarization-{now}.json', 'w') as f:
        #    json.dump(context, f, indent="\t")
    else:
        context = data
    
    # answer extraction
    ca = answer_extraction(args.is_biomedical, context['data'], args.do_summary)
    #now = datetime.now()
    #with open(f'{output_file}/answer-extraction-{now}.json', 'w') as f:
    #    json.dump(ca, f, indent="\t")

    # question generation
    qg_tokenizer = AutoTokenizer.from_pretrained(args.qg_model_name_or_path)
    qg_model = AutoModelForSeq2SeqLM.from_pretrained(args.qg_model_name_or_path, use_cache=True)
    qg_pipe = pipeline('text2text-generation', model=qg_model, tokenizer=qg_tokenizer, device=args.device)

    cqa = question_generation(qg_pipe, ca['data'], args.batch_size, args.qg_min_length, args.qg_max_length)

    # iterative filtering
    qa_tokenizer = AutoTokenizer.from_pretrained(args.qa_model_name_or_path)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(args.qa_model_name_or_path)
    qa_pipe = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer, device=args.device)

    filtered_cqa = iterative_filtering(cqa['data'], qg_pipe, qa_pipe, args.batch_size)

    # saving filtered dataset
    save_data(args.output_file, filtered_cqa)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # paths and general setups
    parser.add_argument('--data_file', required=True, type=str)
    parser.add_argument('--output_file', required=True, type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--doc_limit', default=-1, type=int,
                        help="number of documents to process in the input corpus. use -1 if you want to process all documents.")
    parser.add_argument('--is_biomedical', default=False, action="store_true")
    
    # summarization
    parser.add_argument('--do_summary', default=False, action="store_true")
    parser.add_argument('--summary_min_length', default=64, type=int)
    parser.add_argument('--summary_max_length', default=128, type=int)
    parser.add_argument('--summary_model_name_or_path', default='facebook/bart-large-cnn', type=str)

    # question generation
    parser.add_argument('--qg_min_length', default=64, type=int)
    parser.add_argument('--qg_max_length', default=128, type=int)
    parser.add_argument('--qg_model_name_or_path', default='mrm8488/t5-base-finetuned-question-generation-ap', type=str)

    # general domain default setting
    parser.add_argument('--qa_model_name_or_path', default="thatdramebaazguy/roberta-base-squad", type=str,
                        help="use 'dmis-lab/biobert-base-cased-v1.1-squad' as the QA model for the biomedical domain")

    args = parser.parse_args()

    main(args)

