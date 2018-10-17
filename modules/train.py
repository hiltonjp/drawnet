import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from unet import UNet

import gc
import tqdm

from IPython.core.ultrab import AutoFormattedTB
__ITB__ = AutoFormattedTB(mode='Verbose', color_scheme='LightBg', tb_offset=1)

class Trainer:
    """A model for training an image to image model.
    
    Attributes:
        data (Dataset):
        valdata (Dataset):
        objective (Loss):
        optimizer (optim):
        valstep (int):
        sessionPath (str):
        modelName (str):
    """

    def __init__(self, 
                 data=None, 
                 valdata=None, 
                 objective=None, 
                 optimizer=None, 
                 valstep=1, 
                 sessionPath='./',
                 modelName='Model'):

        self._data = data
        self._valdata = valdata
        self._objective = objective
        self._optimizer = optimizer
        self._valstep = valstep
        self._sessionPath = sessionPath

        if sessionPath == None:
            self._session = {
                'metrics': {
                    'train_loss':[]
                    'val_loss':[]
                },
                'weights':None,
                'epoch':0,
                'name':modelName + '-epoch-0',
                'model':modelName
            }
        else:
            self.loadSession(sessionPath, modelName)

    ###########################################################################
    # Setters                                                                 #
    ###########################################################################

    def setData(self, data):
        self._data = data

    def setValData(self, valdata):
        self._valdata = valdata

    def setObjective(self, objective):
        self._objective = objective

    def setOptimizer(self, optimizer):
        self._optimizer = optimizer

    def setValstep(self, valstep):
        self._valstep = valstep

    def setSessionPath(self, path):
        self._sessionPath = path

    def setModelName(self, modelName):
        self.session['model'] = modelName
        self.session['name'] = modelName + '-epoch-' + self.session['epoch']

    ###########################################################################
    # Training Methods                                                        #
    ###########################################################################

    def train(self, 
              model,
              epochs):
        """Train a model on a set of data.
        
        Args:
            model (nn.Module):  A torch module
            epochs (int):       The number of epochs to train for

        Returns:
            (dict): a dictionary containing --
                - metrics['train_loss']
                - metrics['val_loss']
        """

        if self._session['weights'] is not None:
            model.load_state_dict(self._session['weights'])

        try:
            gc.enable()
            gc.collect()

            while self._session['epoch'] <= epochs:
                losses = []
                accuracies = []
                loop = tqdm(total=len(data), position=0)

                for batch, (x, y_true) in enumerate(data):
                    x, y_true = x.cuda(async=True), y_true.cuda(async=True)
                    y_pred = model(x)

                    # caculate and store accuracy
                    loss = self._objective(y_pred, y_true)
                    loss.backward()
                    losses.append(loss.item())

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    # loop management
                    loop.set_description(
                        'epoch: {}, loss: {:.4}, accuracy: {:.3f}'
                        .format(epoch, loss.item(), accuracy)
                    )
                    loop.update(1)
                    gc.collect()


                # train averaging
                self._session['metrics']['train_loss'].append(np.mean(losses))

                # validation
                if self._session['epoch'] % self._valstep == 0:
                    val_loss = self.validate(model)
                    self._session['metrics']['val_loss'].append(val_loss)
                
                self._session['epoch'] += 1

                # epoch management
                self.saveSession(model)

                loop.close()

            return metrics
        except:
            __ITB__()
    

    def validate(self, model):
        """Perform validation on a model in training.
        
        Args:
            model (nn.Module): A pytorch model
        
        Returns:
            float: The validation loss
        
        """
        with torch.no_grad():
            predictions = [(model(x.cuda(async=True)), y.cuda(async=True)) for x, y in data]
            val_loss = np.mean([
                self.objective(y_pred, y_true.long()).item() for y_pred, y_true in predictions
            ])

        return val_loss
    
    ###########################################################################
    # Session Management                                                      #
    ###########################################################################

    def saveSession(self, model):
        """Save a training session.
        
        Args:
            model (nn.Module):  A torch model
            metrics (dict):     The set of metrics to save alongside the session
            path (str):         The path to save the weights and session data to.
        """       
        sesh_file = os.path.join(self._sessionPath, self._session['name']+'.sesh')
        self._session['weights'] = model.state_dict()
        with open(sesh_file, 'wb') as f:
            pickle.dump(self._session, f, pickle.HIGHEST_PROTOCOL)


    def loadSession(self, path, name):
        """Load a saved training session.

        Notes:
            Session information loads a dictionary containing:
                - session['weights'] (str):  the path to the weights
                - session['metrics'] (dict): a dictionary of metrics
                - session['epoch'] (int):    The current training epoch
                - session['name'] (str):     The name of the session

        Args:
            path (str): Where the session is saved
            name (str): The name of the model session (modeltype-epoch-#)

        """
        with open(os.path.join(path, name) + '.sesh', 'rb') as f:
            self._session = pickle.load(f)
            self.setSessionPath(path)
            


