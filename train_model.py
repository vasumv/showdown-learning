from deepx.nn import *
from deepx.loss import *
from deepx.optimize import *
from tqdm import tqdm

from showdown_parser import ReplayDatabase, parse_log
from showdown_rl import Converter

def predict(experience):

    x, _ = converter.encode_experience(experience)
    probs = net.predict(x[None])[0]
    predictions = probs.argsort()[::-1][:5]

    print experience[0], experience[1]
    print

    for i, prediction in enumerate(predictions):
        print "Prediction[%u]: " % i,
        orig = prediction
        if prediction >= converter.move_index:
            prediction -= converter.move_index
            print "Switch(%s) %.3f" % (converter.poke_backward_mapping[prediction], probs[orig])
        else:
            print "Move(%s) %.3f" % (converter.move_backward_mapping[prediction], probs[orig])

def train(iters):

    idx = np.arange(len(experiences))
    avg_loss = None
    for i in xrange(iters):
        es = experiences[np.random.permutation(idx)[:200]]
        X, y = zip(*[converter.encode_experience(e) for e in es])
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        loss = adam.train(X, y, 0.001)
        if avg_loss is None:
            avg_loss = loss
        else:
            avg_loss = avg_loss * 0.90 + 0.10 * loss
        print "Loss[%u]: %.3f [%.3f]" % (i, loss, avg_loss)

if __name__ == "__main__":
    #r = ReplayDatabase('replays_all.db')
    #experiences = []
    #for id, replay_id, log in tqdm(r.get_replays(limit=10000)):
        #try:
            #for experience in parse_log(log):
                #experiences.append(experience)
        #except:
            #pass
    #converter = Converter()
    #converter.learn_encodings(experiences)
    #experiences = np.array(experiences)
    ##np.random.seed(100)
    ##experiences = np.random.permutation(experiences)[:10000]

    #net = Vector(converter.get_input_dimension()) >> Repeat(Tanh(1500), 2) >> Softmax(converter.get_output_dimension())

    adam = Adam(net >> CrossEntropy())
