''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve FRB_model_slider.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
from scipy.signal import convolve
import sys,os
MIN_FLOAT = sys.float_info[3]

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox,column
from bokeh.models import ColumnDataSource, Div
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure



def get_mad(ts):
    return np.median(np.abs(ts - np.median(ts)))

def normalise(ts):
    return ts/(1.4826*get_mad(ts))


def Gaussian1D(x,sig,x0):
    return np.exp(-(x-x0)*(x-x0)/(2*sig*sig + MIN_FLOAT))

def linear(x,a,b):
    return a*x + b

def exp_decay(x,tau,x0):
    res = np.zeros(len(x)) + MIN_FLOAT
    #res[x <= x0] = MIN_FLOAT
    res[x > x0] = np.exp(-(x[x>x0]-x0)/(tau+MIN_FLOAT))
    return res

def exp_gauss(x,x0,amp,sig,tau,eps):
    gx0 = np.mean(x)
    g = Gaussian1D(x,sig,gx0)
    ex = exp_decay(x,tau,x0)
    conv = convolve(g,ex,"same")
    conv /= np.max(conv) + MIN_FLOAT
    return amp*conv + eps

def exp_gauss_3(x,x1,amp1,sig1,tau1,
               x2,amp2,sig2,tau2,
               x3,amp3,sig3,tau3):
    g1 = exp_gauss(x,x1,amp1,sig1,tau1,0)
    g2 = exp_gauss(x,x2,amp2,sig2,tau2,0)
    g3 = exp_gauss(x,x3,amp3,sig3,tau3,0)
    return g1 + g2 + g3

def lnlike(theta, x, y):
    model = exp_gauss_3(x,*theta)
#    inv_sig = 1./(model**2)
    chisqr = -0.5*(np.sum((y-model)**2))
    return chisqr


y_data = np.load("./tseries.npy")
sampling_time, sampling_unit = 0.01024, "msec"

x = np.arange(len(y_data)) * sampling_time


#x0,amp,sig,tau,eps
param_names = ['x1','amp1','sig1','tau1',
              'x2','amp2','sig2','tau2',
              'x3','amp3','sig3','tau3']
p0 = [5.0, 22., 0.08, 0.15,
     6.15, 7., 0.08, 0.15,
     6.9, 12., 0.08, 0.15]

lower_bounds = [4.85, 1, 0.001, 0.0001,
               6.0, 1., 0.001, 0.0001,
               6.75, 1., 0.001, 0.0001]
upper_bounds = [5.15, 30, 0.3, 0.6,
               6.3,  20, 0.3, 0.6,
               7.0, 20, 0.3, 0.6]

bounds = (lower_bounds, upper_bounds)


y = exp_gauss_3(x, *p0)

lnlike_val = lnlike(p0, x, y_data)
lnlikebox = Div(text="",width=800)
lnliketext = open(os.path.join(os.path.dirname(__file__), "lnlike.html")).read()
lnlikebox.text = lnliketext + "<h1> lnlikelihood:</p>%.3f </h1>" %lnlike_val

# Set up data
source = ColumnDataSource(data=dict(x=x, y=y, res=(y-y_data)))


# Set up plot
plot = figure(plot_height=500, plot_width=700, title="FRB181017",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, np.max(x)], y_range=[-3, 1.2*np.max(y_data)])

plot_residual = figure(plot_height=200, plot_width=700, title="Residuals",
        tools="crosshair,pan,reset,save,wheel_zoom")

plot.line('x', 'y', source=source, line_width=2, line_alpha=0.6, line_color='red')
plot.line(x, y_data, line_width=2, line_alpha=0.6, line_color='blue')

plot_residual.line('x', 'res', source=source, line_width=2, line_alpha=0.6, line_color='black')


# Set up widgets
text = TextInput(title="title", value='FRB181017')
sliders = [Slider(title=param_names[i], value=p0[i], start=bounds[0][i], end=bounds[1][i], step=(bounds[1][i]-bounds[0][i])/1000.) 
        for i in range(len(param_names))]


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):

    # Get the current slider values
    p = [i.value for i in sliders]

    y_new = exp_gauss_3(x, *p)
    lnlike_val = lnlike(p, x, y_data)

    lnlikebox.text = lnliketext + "<h1> lnlikelihood:</p>%.3f </h1>" %lnlike_val
#    print "y_data", np.max(source.data['y_data'])
#    print "y:", source.data['y']

    source.data = dict(x=x, y=y_new, res=y_new-y_data)

for w in sliders:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = widgetbox(text, *sliders)

curdoc().add_root(row(inputs, column(plot,plot_residual), lnlikebox, width=800))
curdoc().title = "Sliders"
