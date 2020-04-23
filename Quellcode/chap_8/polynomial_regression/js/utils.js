 /** Zeichnen der Kurve */
 function drawCurve(x, y) {

    // Extrahiert das maximum von Y
    var maxY = y.max().toInt().dataSync()[0];
    
    var options = {
      fullWidth: true,
      high: maxY,
      low: 0,
      chartPadding: {
        right: 40
      },
      axisX: {
        showGrid: true,
        showLabel: true,
        offset: 0.25
      },
      axisY: {
        showGrid: true,
        showLabel: true,

      },
      showLine: true,
      showPoint: true,
      lineSmooth: false,
    }

    // Initial curve
    data = {
      labels: Array.from(x.dataSync()),
      series: [
        Array.from(y.dataSync()),
      ]
    }

    chart = new Chartist.Line('.ct-chart', data, options);
    chart.update(data);

  }