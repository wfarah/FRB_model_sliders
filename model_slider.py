import numpy as np

from bokeh.layouts import row, widgetbox
from bokeh.models import CustomJS, Slider
from bokeh.plotting import figure, output_file, show, ColumnDataSource



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

def exp_gauss_3(x,x0,amp1,sig1,tau1,eps1,
               dx1,amp2,sig2,tau2,eps2,
               dx2,amp3,sig3,tau3,eps3):
    g1 = exp_gauss(x,x0,amp1,sig1,tau1,eps1)
    g2 = exp_gauss(x,x0+dx1,amp2,sig2,tau2,eps2)
    g3 = exp_gauss(x,x0+dx2,amp3,sig3,tau3,eps3)
    return g1 + g2 + g3


arch = psrchive.Archive_load("./FRB181017_239.97.ar.bp")

arch.dedisperse()
arch.remove_baseline()

data = arch.get_data().squeeze()
coarse_start, coarse_end = 29600,30700
tseries = normalise(data.sum(axis=0))

y_data = tseries[coarse_start:coarse_end]

integ = arch.get_Integration(0)
sampling_time, sampling_unit = integ.get_folding_period()/integ.get_nbin() * 1000, "msec"
channel_width,channel_unit = integ.get_bandwidth()/integ.get_nchan(), "MHz"

x = np.arange(len(data)) * sampling_time


source = ColumnDataSource(data=dict(x=x, y=y))

plot = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

callback = CustomJS(args=dict(source=source), code="""
    var data = source.data;
    var A = amp.value;
    var k = freq.value;
    var phi = phase.value;
    var B = offset.value;
    var x = data['x']
    var y = data['y']
    for (var i = 0; i < x.length; i++) {
        y[i] = B + A*Math.sin(k*x[i]+phi);
    }
    source.change.emit();
""")

amp_slider = Slider(start=0.1, end=10, value=1, step=.1,
                    title="Amplitude", callback=callback)
callback.args["amp"] = amp_slider

freq_slider = Slider(start=0.1, end=10, value=1, step=.1,
                     title="Frequency", callback=callback)
callback.args["freq"] = freq_slider

phase_slider = Slider(start=0, end=6.4, value=0, step=.1,
                      title="Phase", callback=callback)
callback.args["phase"] = phase_slider

offset_slider = Slider(start=-5, end=5, value=0, step=.1,
                       title="Offset", callback=callback)
callback.args["offset"] = offset_slider

layout = row(
    plot,
    widgetbox(amp_slider, freq_slider, phase_slider, offset_slider),
)

output_file("slider.html", title="slider.py example")

show(layout)
