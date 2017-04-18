import tensorflow as tf
import squad_input
import evaluate

from dcn import DCN, Config

def test(model=DCN, config=Config, global_step=10000, filename='dev'):
    with tf.Graph().as_default():

        filename_queue = tf.train.string_input_producer(
                ['data/squad/proto/{}.proto'.format(filename)], num_epochs=1)

        batch_inputs = squad_input.batch_inputs(filename_queue, train=False)

        embedding, ivocab, dataset = squad_input.load_data(filename)

        # initialize model
        model = model(config, batch_inputs, embedding)

        with tf.Session() as sess:
        
            test_writer = tf.summary.FileWriter('log/' + config.save_path + 'test', sess.graph)

            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            restorer = tf.train.Saver()
            restorer.restore(sess, 'weights/' + config.save_path + '-{}'.format(global_step))

            predictions = {}
            try:
                step = 0
                while True:
                    out = sess.run([model.loss, model.pred, model.merged] + batch_inputs)
                    loss, preds, summary = out[:3]
                    test_writer.add_summary(summary, global_step+step)
                    out = out[3:]
                    squad_input.add_preds(predictions, preds, out[2], out[-1], ivocab)
                    step += 1
            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)
                return evaluate.evaluate(dataset, predictions)


if __name__ == '__main__':
    model = DCN
    config = Config()
    test(model, config)


