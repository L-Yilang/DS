let currentSlide = 0;
const totalSlides = 7; // 0 to 6

function goToSlide(index) {
    if (index < 0 || index >= totalSlides) return;
    
    document.querySelectorAll('.slide').forEach((slide, i) => {
        if (i === index) {
            slide.classList.add('active');
        } else {
            slide.classList.remove('active');
        }
    });
    
    currentSlide = index;

    // 如果切换到结果页面，初始化或更新图表
    if (index === 5) {
        setTimeout(initChart, 300); // 延迟等待动画完成，确保容器有尺寸
    }
}

function nextSlide() {
    goToSlide(currentSlide + 1);
}

function prevSlide() {
    goToSlide(currentSlide - 1);
}

// 监听键盘左右键
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') {
        nextSlide();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        prevSlide();
    }
});

// 监听鼠标点击背景跳转下一页
document.addEventListener('click', (e) => {
    // 如果点击的是按钮、链接、下拉框、或者导航菜单，不触发全局跳转
    const isInteractive = e.target.closest('button, a, select, li, .nav-list, .algo-tabs, .chart-controls, iframe');
    if (!isInteractive) {
        nextSlide();
    }
});

// 算法 Tab 切换
function showAlgo(algoId) {
    document.querySelectorAll('.algo-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    document.getElementById(algoId).classList.add('active');
    event.currentTarget.classList.add('active');
}

// ECharts 结果对比图表
let myChart = null;

// 模拟数据
const mockData = {
    small: {
        score: [1500, 1600, 1800, 2100, 2200, 2300, 2500],
        completion: [70, 75, 82, 90, 92, 95, 100],
        distance: [500, 480, 450, 400, 380, 390, 350]
    },
    medium: {
        score: [3000, 3200, 3600, 4200, 4400, 4600, 5000],
        completion: [65, 70, 78, 88, 90, 93, 100],
        distance: [1200, 1150, 1050, 950, 900, 920, 850]
    },
    large: {
        score: [6000, 6500, 7200, 8500, 8800, 9200, 10000],
        completion: [60, 65, 75, 85, 88, 90, 100],
        distance: [3000, 2900, 2700, 2400, 2300, 2350, 2100]
    }
};

const strategies = ['Nearest Task', 'Max Weight', 'Time First Bundle', 'ALNS', 'Genetic Hyper', 'MAPPO', 'Gurobi (Optimal)'];

function initChart() {
    if (!myChart) {
        const chartDom = document.getElementById('resultChart');
        if (chartDom.clientWidth === 0) return; // 如果容器还不可见
        myChart = echarts.init(chartDom);
    }
    updateChart();
}

function updateChart() {
    if (!myChart) return;

    const scale = document.getElementById('scaleSelect').value;
    const metric = document.getElementById('metricSelect').value;
    
    const data = mockData[scale][metric];
    
    let yAxisName = '';
    let seriesName = '';
    let color = '';

    if (metric === 'score') {
        yAxisName = '总得分';
        seriesName = 'Score';
        color = '#22D3EE';
    } else if (metric === 'completion') {
        yAxisName = '完成率 (%)';
        seriesName = 'Completion Rate';
        color = '#A78BFA';
    } else {
        yAxisName = '总里程';
        seriesName = 'Distance';
        color = '#F472B6';
    }

    const option = {
        title: {
            text: `${scale.toUpperCase()} 规模下的 ${yAxisName} 对比`,
            left: 'center',
            textStyle: { color: '#E0F2FE' }
        },
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' }
        },
        grid: {
            left: '5%',
            right: '5%',
            bottom: '10%',
            containLabel: true
        },
        xAxis: {
            type: 'category',
            data: strategies,
            axisLabel: { interval: 0, rotate: 15, color: '#94A3B8' }
        },
        yAxis: {
            type: 'value',
            name: yAxisName,
            nameTextStyle: { color: '#94A3B8' },
            axisLabel: { color: '#94A3B8' }
        },
        series: [
            {
                name: seriesName,
                type: 'bar',
                data: data,
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: color },
                        { offset: 1, color: '#A78BFA' }
                    ]),
                    borderRadius: [5, 5, 0, 0]
                },
                label: {
                    show: true,
                    position: 'top',
                    color: '#E0F2FE'
                },
                animationDuration: 1500,
                animationEasing: 'cubicOut'
            }
        ]
    };

    myChart.setOption(option);
}

// 窗口大小改变时重绘图表
window.addEventListener('resize', () => {
    if (myChart) {
        myChart.resize();
    }
});
