import tensorflow as tf
import numpy as np
def read_table(filename_queue):
    batch_size = 128
    reader = tf.TableRecordReader(csv_delimiter=';', num_threads=8, capacity=8*batch_size)
    key, value = reader.read_up_to(filename_queue, batch_size)
    values = tf.train.batch([value], batch_size=batch_size, capacity=8*capacity, enqueue_many=True, num_threads=8)
    record_defaults = [[1.0], [""], [""], [""], [""], [""]]
    feature_size = [1322,30185604,43239874,5758226,41900998]
    col1, col2, col3, col4, col5, col6 = tf.decode_csv(values, record_defaults=record_defaults, field_delim=';')
    outmatrix = tf.trans_csv_to_dense(['2,3,5','2,6,7,7','0,9,3'],6)
    col2 = tf.trans_csv_kv2dense(col2, feature_size[0])
    col3 = tf.trans_csv_id2sparse(col3, feature_size[1])
    col4 = tf.trans_csv_id2sparse(col4, feature_size[2])
    col5 = tf.trans_csv_id2sparse(col5, feature_size[3])
    col6 = tf.trans_csv_id2sparse(col6, feature_size[4])
    return [col1, col2, col3, col4, col5, col6]
if __name__ == '__main__':    
    tf.app.flags.DEFINE_string("tables", "", "tables")
    tf.app.flags.DEFINE_integer("num_epochs", 1000, "number of epoches")
    FLAGS = tf.app.flags.FLAGS
    table_pattern = FLAGS.tables
    num_epochs = FLAGS.num_epochs
    filename_queue = tf.train.string_input_producer(table_pattern, num_epochs)
    train_data = read_table(filename_queue)
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1000):
            sess.run(train_data)
        coord.request_stop()
        coord.join(threads)