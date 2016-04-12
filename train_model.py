from deepx.nn import *
from deepx.loss import *
from deepx.optimize import *
from tqdm import tqdm

from showdown_parser import ReplayDatabase, parse_log
from showdown_rl import Converter

def correct_poke(poke):
    return poke.split(',')[0]

def predict(experience):
    state, action, _, _ = experience

    x = converter.encode_state(state)

    probs = net.predict(x[None])[0]
    predictions = probs.argsort()[::-1][:5]

    print "Matchup: %s[%.2f] vs %s[%.2f]" % (correct_poke(state.get_primary(0).name),
                                         state.get_health(0),
                                         correct_poke(state.get_primary(1).name),
                                         state.get_health(1))
    print "My team: %s" % ', '.join(["%s[%f]" % (correct_poke(p.get_name()), p.health) for p in state.get_team(0)[1:]])
    print "Their team: %s" % ', '.join([correct_poke(p.get_name()) for p in state.get_team(1)[1:]])
    print
    print "My action: %s" % experience[1]
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
    exs = np.random.permutation(experiences)
    for i in xrange(iters):
        ix = np.random.choice(idx)
        es = exs[ix: ix + 512]
        X, y = zip(*[(converter.encode_state(e[0]), converter.encode_action(e[1])) for e in es])
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        loss = adam.train(X, y, 0.001)
        if avg_loss is None:
            avg_loss = loss
        else:
            avg_loss = avg_loss * 0.90 + 0.10 * loss
        print "Loss[%u]: %.3f [%.3f]" % (i, loss, avg_loss)

if __name__ == "__main__":
    r = ReplayDatabase('replays_all.db')
    experiences = []
    for id, replay_id, log in tqdm(r.get_replays(limit=50000)):
        try:
            for experience in parse_log(log):
                experiences.append(experience)
        except:
            print "Failed on replay:", replay_id
            # import traceback
            # traceback.print_exc()
    converter = Converter()
    converter.learn_encodings(experiences)
    experiences = np.array(experiences)

    net = Vector(converter.get_input_dimension()) >> Repeat(Tanh(1000), 2) >> Softmax(converter.get_output_dimension())

    adam = Adam(net >> CrossEntropy())
