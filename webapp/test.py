import os
import mne
import pickle as pkl
pth_modeling = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'assets\\modeling'))

epochs = mne.read_epochs(pth_modeling + '\\epochs-epo.fif')
evoked = epochs.average()
evoked.save(pth_modeling + '\\evoked-ave.fif')

test = mne.Evoked(pth_modeling + '\\evoked-ave.fif')
print(test)
