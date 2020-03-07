
from Plot_Scene import Plot_Scene


def test(room,mic_label,azi_tar,n_inter,model):

    azi_theta = 1

    filter_gpu = Filter_GPU(0)
    cues, azi_gt_all = gen_test_sample(room, mic_label, azi_tar,
                                       n_inter, filter_gpu)

    azi_est = model.locate(cues)

    azi_diff = np.min(np.abs(np.subtract(azi_gt_all.reshape([-1,1]),azi_est)),
                      axis=0)

    n_frame = azi_est.shape[0]
    n_frame_correct = np.count_nonzero(azi_diff<azi_theta)
    cp = n_frame_correct/n_frame

    scene = Plot_Scene(src_azi_all=azi_gt_all,mic_label=mic_label)
    scene.plot_est(azi_est)
    scene.savefig('images/{}_{}_{}_{}_scence.png'.format(room,mic_label,
                                                         azi_tar,n_inter))

    fs = 44100
    frame_len = 20e-3
    shift_len = 10e-3
    fig,ax = plt.subplots(3,1,sharex=True,tight_layout=True)
    for record_i,record in enumerate(record_all):
        ax[0].plot(np.arange(record.shape[0])/fs,record[:,0],
                   label='source {}'.format(record_i))
        ax[1].plot(np.arange(record.shape[0])/fs,record[:,1],
                   label='source {}'.format(record_i))
    ax[0].legend()
    ax[0].set_title('L')
    ax[1].legend()
    ax[1].set_title('R')
    ax[2].plot(np.arange(n_frame)*shift_len+int(frame_len/2),azi_est)
    for azi_gt_i,azi_gt in enumerate(azi_gt_all):
        ax[2].plot([0,(n_frame-1)*shift_len+int(frame_len/2)],[azi_gt,azi_gt],
                   label='source {}'.format(azi_gt_i))
    ax[2].legend()
    ax[2].set_ylim([7,29])
    ax[2].set_title('{:.2f}'.format(cp))
    plot_tools.savefig(fig,name='{}_{}_{}_{}_result.png'.format(room,mic_label,
                                                            azi_tar,n_inter))


