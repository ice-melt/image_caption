# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import math
import tensorflow as tf
import configuration
import inference_wrapper
import baidu_trans
from inference_utils import caption_generator
from inference_utils import vocabulary
from flask import Flask, render_template, request, jsonify
#from flask import redirect, url_for, make_response
from werkzeug.utils import secure_filename
from datetime import timedelta

# 是否保持用户上传的图片
IS_CACHE_IMG = False
TRANS_Lang = 'zh'#[zh,en]
# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG'])

def initGraph():
    tf.logging.set_verbosity(tf.logging.INFO)
    displayconfig = configuration.DisplayConfig()
    checkpoint_path = displayconfig.checkpoint_path
    vocab_file = displayconfig.vocab_file

    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    v = vocabulary.Vocabulary(vocab_file)

    s = tf.Session(graph=g)
    # Load the model from checkpoint.
    restore_fn(s)
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    return s,v,caption_generator.CaptionGenerator(model, v)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
sess,vocab,generator = initGraph()
trans = baidu_trans.trans()

@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():

    option = {'en': '', 'zh': ''}
    msg = ""

    # noinspection PyBroadException
    try:
        if request.method == 'POST':

            transFlag = request.form['language'] == 'zh'
            if transFlag:
                option = {'en': '', 'zh': 'selected'}
            else:
                option = {'en': 'selected', 'zh': ''}

            if 'file' not in request.files:
                msg = "请选择图片文件上传"
                return render_template('upload.html', message=msg,o=option)

            f = request.files['file']

            if not (f and allowed_file(f.filename)):
                msg = "请检查上传的图片类型，仅限于png、PNG、jpg、JPG"
                return render_template('upload.html', message=msg,o=option)
                # return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG"})

            basepath = os.path.dirname(__file__)  # 当前文件所在路径
            uploadfile = secure_filename(f.filename) if IS_CACHE_IMG else 'test.jpg'
            upload_path = os.path.join(basepath, 'static/images',uploadfile)  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
            test_img_path = os.path.join(basepath, 'static/images', 'test.jpg')
            f.save(upload_path)

            # 使用Opencv转换一下图片格式和名称，统一成jpg格式（3通道的）
            img = cv2.imread(upload_path)
            cv2.imwrite(test_img_path, img)

            with tf.gfile.GFile(test_img_path, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)

            # 每张图片都有三句话描述，所以 sentences 的长度是3
            sentences = {}
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                if transFlag:
                    sentence = trans.baidu_fanyi(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                sentences[i] = {}
                sentences[i]['word'] = ("%s (p=%f)" % ( sentence, math.exp(caption.logprob)))
            #print(sentences)
            return render_template('upload_ok.html', s=sentences,o=option)
    except:
        #上传的文件是其他文件改成jpg格式会报错
        msg = 'something error,please make sure the uploaded file is in JPG format!'
        return render_template('upload.html', message=msg,o=option)
    return render_template('upload.html', message=msg,o=option)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=True)