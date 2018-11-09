# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import tensorflow as tf
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from datetime import timedelta


#checkpoint_path = "data/train_output/model.ckpt-1000"
checkpoint_path = "data/output_mscoco/model.ckpt-200000"
#vocab_file = "data/word_counts.txt"
vocab_file = "data/word_counts_mscoco.txt"

tf.logging.set_verbosity(tf.logging.INFO)

g = tf.Graph()
with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),checkpoint_path)
g.finalize()

# Create the vocabulary.
vocab = vocabulary.Vocabulary(vocab_file)

sess = tf.Session(graph=g)
# Load the model from checkpoint.
restore_fn(sess)
# Prepare the caption generator. Here we are implicitly using the default
# beam search parameters. See caption_generator.py for a description of the
# available beam search parameters.
generator = caption_generator.CaptionGenerator(model, vocab)


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    print('进入了upload')
    if request.method == 'POST':
        f = request.files['file']
        print('wtf f----',f)
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        print('之前的 upload_path 是',upload_path)
        filenames = tf.gfile.Glob(upload_path)
        filename = filenames[0]
        # 得到上传图片进行 inference
        print('the filename is ',filename)
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = generator.beam_search(sess, image)
        print(captions)
        print("Captions for image %s:" % os.path.basename(filename))

        # 每张图片都有三句话描述，所以 sentences 的长度是3
        sentences = {}
        for i, caption in enumerate(captions):
            # Ignore begin and end words.
            sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            sentences[i] = ("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
        print(sentences[0])
        return render_template('upload_ok.html', s1=sentences[0])

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8987, debug=True)