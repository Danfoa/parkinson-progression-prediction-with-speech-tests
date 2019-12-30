from regression_models.anfis_packages.anfis import ANFIS
import regression_models.anfis_packages.anfis
from regression_models.anfis_packages import membershipfunction
import numpy as np


class AMFIS:
    def __init__(self, data, y):
        self.data = data
        self.y = y
        self.num_invars = np.size(data, 1)

        mf = self.create_gauss_mfs(data)
        self.mfc = membershipfunction.MemFuncs(mf)
        self.anf = None


    def fit(self):
        self.anf = ANFIS(self.data, self.y, self.mfc)
        self.anf.trainHybridJangOffLine(epochs=10)

    def predict(self, data):

        return regression_models.anfis_packages.anfis.predict(self.anf, data)




    def make_gauss_mfs(self, sigma, mu_list):
        '''Return a list of gaussian mfs, same sigma, list of means'''
        # {'mean':0.,'sigma':1.}
        l = []
        for mu in mu_list:
            l.append(['gaussmf', {'mean' : mu, 'sigma': sigma}])
        # mf = [GaussMembFunc(mu, sigma) for mu in mu_list]
        return l


    def create_gauss_mfs(self, x, num_mfs=3, num_out=2, hybrid=True):
        minvals = np.min(self.data, axis=0)

        maxvals = np.max(self.data, axis=0)
        ranges = np.abs(maxvals) - np.abs(minvals)
        invars = []
        for i in range(self.num_invars):
            sigma = ranges[i] / num_mfs
            mulist = np.linspace(minvals[i], maxvals[i], num_mfs).tolist()
            invars.append(self.make_gauss_mfs(sigma, mulist))

        outvars = ['y{}'.format(i) for i in range(num_out)]
        return invars


class GaussMembFunc(object):
    def __init__(self, mu, sigma):
        super(GaussMembFunc, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def __setitem__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __getitem__(self, arg: object) -> object:
        return GaussMembFunc[arg]

    def forward(self, x):
        val = np.exp(-np.pow(x - self.mu, 2) / (2 * self.sigma**2))
        return val


