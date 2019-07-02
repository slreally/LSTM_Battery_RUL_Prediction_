import matplotlib.pyplot as plt
import datetime

#save_filepath: 'result/single_variable/'
def plot_and_save(title, sequence_length, save_filename,save_filepath, all_y, test_y, train_y, predict_y):
    fig2 = plt.figure()
    plt.xlabel('cycle')
    plt.ylabel('QD')
    # plt.plot(range(sequence_length,sequence_length+len(all_y)),all_y)
    plt.plot(all_y)
    plt.plot(range(sequence_length,sequence_length+len(train_y),1),train_y,'m:')
    plt.plot(range(sequence_length+len(train_y),sequence_length+len(train_y)+len(test_y),1),test_y,'r:')
    plt.plot(range(sequence_length+len(train_y),sequence_length+len(train_y)+len(predict_y),1),predict_y,'g-')
    time = get_time()
    plt.title(title)
    plt.legend(['ground truth','train','test','predict'])
    plt.show()
    save_filename = save_filename + "_" + str(time)
    plt.savefig(save_filepath + save_filename + '.png')
    plt.close(fig2)

def get_time():
    time = datetime.datetime.now().strftime('%m-%d-%H-%R-%S')
    return time