(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // Font config: Arial bold, 20px+ per project convention
  var FONT = 'Arial';
  var labelStyle = {
    fontFamily: FONT,
    fontSize: 20,
    fontWeight: 'bold',
    color: ink
  };
  var tickStyle = {
    fontFamily: FONT,
    fontSize: 14,
    fontWeight: 'bold',
    color: muted
  };

  var el = document.getElementById('chart-pi-compare');
  if (!el) return;

  var chart = echarts.init(el, null, { renderer: 'svg' });

  var categories = [
    'Agent Loop\n成熟度',
    '会话转录',
    '上下文压缩',
    '运行时引导',
    '多模态支持',
    '工具生态',
    '通信渠道',
    'UI 丰富度'
  ];

  chart.setOption({
    animation: false,
    backgroundColor: 'transparent',
    color: [accent, accent2],
    legend: {
      data: ['Huginn', 'pi-mono'],
      top: 10,
      textStyle: labelStyle,
      itemWidth: 20,
      itemHeight: 14,
      itemGap: 30
    },
    grid: {
      left: 160,
      right: 40,
      top: 60,
      bottom: 30
    },
    xAxis: {
      type: 'value',
      max: 5,
      min: 0,
      interval: 1,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { fontFamily: FONT, fontSize: 14, fontWeight: 'bold', color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    yAxis: {
      type: 'category',
      data: categories,
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: {
        fontFamily: FONT,
        fontSize: 14,
        fontWeight: 'bold',
        color: ink,
        lineHeight: 18
      }
    },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      appendToBody: true,
      textStyle: { fontFamily: FONT, fontSize: 14, fontWeight: 'bold' },
      formatter: function(params) {
        var s = params[0].name.replace('\n', ' ') + '<br/>';
        params.forEach(function(p) {
          s += p.marker + ' ' + p.seriesName + ': ' + p.value + '/5<br/>';
        });
        return s;
      }
    },
    series: [
      {
        name: 'Huginn',
        type: 'bar',
        data: [3, 2, 1, 1, 4, 5, 4, 4],
        barWidth: '35%',
        itemStyle: {
          color: accent,
          borderRadius: [0, 4, 4, 0]
        },
        label: {
          show: true,
          position: 'right',
          fontFamily: FONT,
          fontSize: 14,
          fontWeight: 'bold',
          color: ink
        }
      },
      {
        name: 'pi-mono',
        type: 'bar',
        data: [5, 5, 5, 4, 2, 3, 2, 4],
        barWidth: '35%',
        itemStyle: {
          color: accent2,
          borderRadius: [0, 4, 4, 0]
        },
        label: {
          show: true,
          position: 'right',
          fontFamily: FONT,
          fontSize: 14,
          fontWeight: 'bold',
          color: ink
        }
      }
    ]
  });

  window.addEventListener('resize', function() { chart.resize(); });
})();
