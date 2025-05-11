import * as fs from 'fs';
import * as path from 'path';
import * as d3 from 'd3';
import { JSDOM } from 'jsdom';

// --- Data Structures ---
interface Metric {
    loss: number;
    step: number;
    accuracy?: number;
    klDivergence?: number;
    alphaFirstParams?: number[];
    alphaSecondParams?: number[];
    [key: string]: number | number[] | undefined;
}

type ExperimentData = [string, Metric[]][]; // [ExperimentName, MetricsOverSteps[]]

// Define available metrics
const SCALAR_METRICS = ['loss', 'accuracy', 'klDivergence'] as const;
const ARRAY_METRICS = ['alphaFirstParams', 'alphaSecondParams'] as const;
type ScalarMetric = typeof SCALAR_METRICS[number];
type ArrayMetric = typeof ARRAY_METRICS[number];
type AnyMetric = ScalarMetric | ArrayMetric;

// Type for transformed data points used for plotting
type PlotPoint = { step: number; value: number };
// Type for data structure passed to the generic plotting function
// [LegendKey (string or number), PlotPoints[]]
type PlotData = [string | number, PlotPoint[]][];


// --- Plotting Configuration ---
const config = {
    width: 650,
    height: 400,
    margin: { top: 50, right: 180, bottom: 60, left: 60 }, // Increased top margin for title
    axisLabelFontSize: '12px',
    tickFontSize: '10px',
    titleFontSize: '14px', // Slightly smaller title for potentially long auto-generated titles
    legendFontSize: '10px',
    lineWidth: 1.5,
    fontFamily: 'Helvetica Neue, Helvetica, Arial, sans-serif',
    gridColor: '#e0e0e0',
    axisColor: '#333333',
    labelColor: '#000000',
};


// --- Generic Plotting Function ---
/**
 * Creates a NeurIPS-style line plot from pre-processed data.
 * Allows specifying Y-axis limits which act as bounds on the data range.
 * @param plotData Data formatted as [LegendKey, {step, value}[]][].
 * @param outputFilePath Path to save the generated SVG file.
 * @param plotTitle Title for the plot.
 * @param xAxisLabel Label for the X-axis.
 * @param yAxisLabel Label for the Y-axis.
 * @param yMinBound Optional minimum boundary for the Y-axis. Final min = max(yMinBound, dataMin).
 * @param yMaxBound Optional maximum boundary for the Y-axis. Final max = min(yMaxBound, dataMax).
 */
function createPlot(
    plotData: PlotData,
    outputFilePath: string,
    plotTitle: string,
    xAxisLabel: string,
    yAxisLabel: string,
    yMinBound?: number, // Optional Y-axis minimum boundary
    yMaxBound?: number  // Optional Y-axis maximum boundary
): void {
    // --- JSDOM Setup ---
    const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>');
    const body = d3.select(dom.window.document.body);

    // --- SVG Canvas Setup ---
    const plotWidth = config.width - config.margin.left - config.margin.right;
    const plotHeight = config.height - config.margin.top - config.margin.bottom;
    const svg = body.append('svg')
        .attr('width', config.width)
        .attr('height', config.height)
        .attr('xmlns', 'http://www.w3.org/2000/svg')
        .style('font-family', config.fontFamily)
        .style('background-color', 'white');
    svg.append('rect').attr('width', '100%').attr('height', '100%').attr('fill', 'white');
    const plotArea = svg.append('g')
        .attr('transform', `translate(${config.margin.left},${config.margin.top})`);

    // Check if data is empty after potential filtering upstream
    if (plotData.length === 0) {
        console.warn(`⚠️ Warning: No data provided to createPlot function for ${outputFilePath}. Plot will be empty.`);
        svg.append("text").attr("x", config.width / 2).attr("y", config.height / 2).attr("text-anchor", "middle")
            .text("No data to plot.");
        try {
            fs.mkdirSync(path.dirname(outputFilePath), { recursive: true });
            fs.writeFileSync(outputFilePath, body.html());
        } catch (err) { console.error(`❌ Error writing empty SVG:`, err); }
        return;
    }

    // --- Scales ---
    const allSteps = plotData.flatMap(([_, points]) => points.map(p => p.step));
    const allYValues = plotData.flatMap(([_, points]) => points.map(p => p.value));

    // Determine X domain (no manual override for X)
    let xDataDomain = d3.extent(allSteps) as [number, number];
    if (!xDataDomain) xDataDomain = [0, 1];
    else if (xDataDomain[0] === xDataDomain[1]) xDataDomain = [xDataDomain[0] - 0.5, xDataDomain[1] + 0.5];
    const xScale = d3.scaleLinear().domain(xDataDomain).range([0, plotWidth]).nice();

    // --- Determine Y domain applying bounds ---
    let yDomain: [number, number];
    let applyNiceToY = true; // Apply .nice() only if no bounds were effectively used or if bounds were invalid

    // 1. Get data extent
    const dataYExtent = d3.extent(allYValues) as [number, number];
    let dataMin = dataYExtent ? dataYExtent[0] : 0;
    let dataMax = dataYExtent ? dataYExtent[1] : 1;

    // Handle case where data has zero range or no data
    if (dataYExtent && dataMin === dataMax) {
        // Add padding if only one data point value exists
        dataMin = dataMin - Math.abs(dataMin * 0.1 || 0.5);
        dataMax = dataMax + Math.abs(dataMax * 0.1 || 0.5);
    } else if (!dataYExtent) { // Handle case where there's no data at all
        dataMin = 0;
        dataMax = 1;
    }


    // 2. Calculate effective min and max based on bounds
    let effectiveMin = dataMin;
    let effectiveMax = dataMax;
    let boundApplied = false; // Track if any user bound was applied

    if (yMinBound !== undefined) {
        effectiveMin = Math.max(yMinBound, dataMin); // Correct: max(limit, data)
        boundApplied = true;
    }
    if (yMaxBound !== undefined) {
        effectiveMax = Math.min(yMaxBound, dataMax); // Correct: min(limit, data)
        console.log(effectiveMax);
        boundApplied = true;
    }

    // 3. Check if the resulting domain is valid (min < max)
    if (effectiveMin >= effectiveMax) {
        console.warn(`⚠️ Warning: Effective y-min (${effectiveMin}) is not less than effective y-max (${effectiveMax}) after applying bounds (--y-min=${yMinBound}, --y-max=${yMaxBound}). Reverting to automatic Y-axis limits based on data extent for ${outputFilePath}.`);
        // Revert to data extent (already handled single point case above)
        yDomain = [dataMin, dataMax];
        applyNiceToY = true; // Apply nice to reverted domain
    } else {
        // Valid domain after applying bounds
        yDomain = [effectiveMin, effectiveMax];
        // Don't apply .nice() if either bound was manually applied and resulted in a valid domain
        applyNiceToY = !boundApplied;
    }
    // --- End of Y-Domain Logic ---


    // Create Y scale
    const yScale = d3.scaleLinear().domain(yDomain).range([plotHeight, 0]);
    if (applyNiceToY) {
        yScale.nice(); // Apply .nice() only if domain wasn't bounded manually or if bounds were invalid
    }


    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
        .domain(plotData.map(([key, _]) => String(key)));

    // --- Axes ---
    const xAxisGenerator = d3.axisBottom(xScale);
    const xAxis = plotArea.append('g').attr('class', 'x-axis').attr('transform', `translate(0,${plotHeight})`)
        .call(xAxisGenerator).style('font-size', config.tickFontSize).style('color', config.axisColor);
    xAxis.select('.domain').remove();
    xAxis.selectAll('.tick line').attr('stroke', config.gridColor).attr('stroke-dasharray', '2,2').attr('y1', -plotHeight).attr('y2', 0);

    const yAxisGenerator = d3.axisLeft(yScale); // Use the final yScale
    const yAxis = plotArea.append('g').attr('class', 'y-axis')
        .call(yAxisGenerator).style('font-size', config.tickFontSize).style('color', config.axisColor);
    yAxis.select('.domain').remove();
    yAxis.selectAll('.tick line').attr('stroke', config.gridColor).attr('stroke-dasharray', '2,2').attr('x1', 0).attr('x2', plotWidth);
    yAxis.selectAll('.tick text').attr('x', -5);

    // --- Axis Labels ---
    svg.append('text').attr('class', 'x-axis-label').attr('x', config.margin.left + plotWidth / 2).attr('y', config.height - config.margin.bottom / 2 + 10)
        .attr('text-anchor', 'middle').style('font-size', config.axisLabelFontSize).style('fill', config.labelColor).text(xAxisLabel);
    svg.append('text').attr('class', 'y-axis-label').attr('transform', 'rotate(-90)').attr('x', -(config.margin.top + plotHeight / 2)).attr('y', config.margin.left / 2 - 15)
        .attr('text-anchor', 'middle').style('font-size', config.axisLabelFontSize).style('fill', config.labelColor).text(yAxisLabel);

    // --- Plot Title ---
    svg.append('text').attr('class', 'plot-title').attr('x', config.width / 2).attr('y', config.margin.top / 2)
        .attr('text-anchor', 'middle').style('font-size', config.titleFontSize).style('fill', config.labelColor).style('font-weight', 'bold').text(plotTitle);

    // --- Line Generator ---
    const lineGenerator = d3.line<PlotPoint>()
        .x(d => xScale(d.step))
        .y(d => yScale(d.value)) // Use the final yScale
        .defined(d => !isNaN(d.value) && d.value !== null)
        .curve(d3.curveLinear);

    // --- Draw Lines ---
    plotArea.append("defs").append("clipPath")
        .attr("id", "clip")
        .append("rect")
        .attr("width", plotWidth)
        .attr("height", plotHeight);

    plotArea.append('g')
        .attr("clip-path", "url(#clip)")
        .selectAll('.line')
        .data(plotData)
        .enter()
        .append('path')
        .attr('class', 'line')
        .attr('d', ([_, points]) => lineGenerator(points))
        .style('fill', 'none')
        .style('stroke', ([key, _]) => colorScale(String(key)))
        .style('stroke-width', config.lineWidth);

    // --- Legend ---
    const legend = svg.append('g').attr('class', 'legend')
        .attr('transform', `translate(${config.width - config.margin.right + 20}, ${config.margin.top})`);
    const legendItems = legend.selectAll('.legend-item')
        .data(plotData)
        .enter()
        .append('g')
        .attr('class', 'legend-item')
        .attr('transform', (d, i) => `translate(0, ${i * 20})`);
    legendItems.append('line').attr('x1', 0).attr('x2', 20).attr('y1', 5).attr('y2', 5)
        .style('stroke', ([key, _]) => colorScale(String(key))).style('stroke-width', config.lineWidth * 1.5);
    legendItems.append('text').attr('x', 25).attr('y', 9)
        .text(([key, _]) => String(key))
        .style('font-size', config.legendFontSize).style('fill', config.labelColor);

    // --- Output SVG to File ---
    try {
        fs.mkdirSync(path.dirname(outputFilePath), { recursive: true });
        const svgOutput = body.html();
        fs.writeFileSync(outputFilePath, svgOutput);
    } catch (err) {
        console.error(`❌ Error writing SVG file to ${outputFilePath}:`, err);
    }
}


// --- Data Preparation & Plotting Functions ---
// Pass yMinBound and yMaxBound down (renamed args for clarity)

/** Plots a scalar metric applying filters */
function plotScalarMetric(
    experimentData: ExperimentData,
    metric: ScalarMetric,
    outputDir: string,
    maxStep: number,
    yMinBound?: number, // Pass down y-limit bounds
    yMaxBound?: number
): boolean {
    console.log(`⏳ Preparing plot for scalar metric "${metric}" (Max Step: ${maxStep === Infinity ? 'None' : maxStep})...`);
    const plotData: PlotData = [];

    for (const [expName, metrics] of experimentData) {
        const points: PlotPoint[] = [];
        for (const m of metrics) {
            if (m.step > maxStep) continue;
            const value = m[metric];
            if (typeof value === 'number' && !isNaN(value)) {
                points.push({ step: m.step, value: value });
            }
        }
        if (points.length > 0) {
            plotData.push([expName, points]);
        }
    }

    if (plotData.length === 0) {
        console.log(`ℹ️ No valid data found for metric "${metric}" within filters. Skipping plot.`);
        return false;
    }

    const outputFilePath = path.join(outputDir, `plot_${metric}.svg`);
    const plotTitle = `${metric.charAt(0).toUpperCase() + metric.slice(1)} vs. Step`;
    const yAxisLabel = metric.charAt(0).toUpperCase() + metric.slice(1);

    createPlot(plotData, outputFilePath, plotTitle, 'Step', yAxisLabel, yMinBound, yMaxBound); // Pass bounds
    console.log(`✅ Scalar plot for "${metric}" saved to ${outputFilePath}`);
    return true;
}


/** Plots a specific layer index applying filters */
function plotLayerMetric(
    experimentData: ExperimentData,
    metric: ArrayMetric,
    layerIndex: number,
    outputDir: string,
    maxStep: number,
    yMinBound?: number, // Pass down y-limit bounds
    yMaxBound?: number
): boolean {
    const plotData: PlotData = [];

    if (layerIndex < 0) {
        console.error(`❌ Invalid layer index: ${layerIndex}. Index must be non-negative.`);
        return false;
    }

    let maxLayersFound = -1;

    for (const [expName, metrics] of experimentData) {
        const points: PlotPoint[] = [];
        for (const m of metrics) {
            if (m.step > maxStep) continue;
            const arr = m[metric];
            if (Array.isArray(arr)) {
                maxLayersFound = Math.max(maxLayersFound, arr.length - 1);
                if (layerIndex < arr.length) {
                    const value = arr[layerIndex];
                    if (typeof value === 'number' && !isNaN(value)) {
                        points.push({ step: m.step, value: value });
                    }
                }
            }
        }
        if (points.length > 0) {
            plotData.push([expName, points]);
        }
    }

    if (layerIndex > maxLayersFound && maxLayersFound !== -1) {
        console.error(`❌ Invalid layer index: ${layerIndex}. Maximum index found in filtered data for "${metric}" is ${maxLayersFound}.`);
        return false;
    }
    if (plotData.length === 0) {
        return false;
    }

    const outputFilePath = path.join(outputDir, `plot_${metric}_layer_${layerIndex}.svg`);
    const plotTitle = `${metric} - Layer ${layerIndex} vs. Step`;
    const yAxisLabel = `${metric}[${layerIndex}]`;

    createPlot(plotData, outputFilePath, plotTitle, 'Step', yAxisLabel, yMinBound, yMaxBound); // Pass bounds
    console.log(`✅ Layer plot for "${metric}" Layer ${layerIndex} saved to ${outputFilePath}`);
    return true;
}


/** Plots all layer indices for experiments applying filters */
function plotExperimentMetric(
    experimentData: ExperimentData,
    metric: ArrayMetric,
    targetExperimentName: string | undefined,
    outputDir: string,
    maxStep: number,
    yMinBound?: number, // Pass down y-limit bounds
    yMaxBound?: number
): number {
    const experimentsToPlot: [string, Metric[]][] = targetExperimentName
        ? experimentData.filter(([name, _]) => name === targetExperimentName)
        : experimentData;

    if (targetExperimentName && experimentsToPlot.length === 0) {
        console.error(`❌ Experiment "${targetExperimentName}" not found in the filtered data.`);
        return 0;
    }
    if (experimentsToPlot.length === 0) {
        console.log(`ℹ️ No experiments to plot (after filtering).`);
        return 0;
    }

    let plotsGenerated = 0;

    for (const [expName, metrics] of experimentsToPlot) {
        const plotData: PlotData = [];
        let maxLayers = 0;

        for (const m of metrics) {
            if (m.step > maxStep) continue;
            const arr = m[metric];
            if (Array.isArray(arr)) {
                maxLayers = Math.max(maxLayers, arr.length);
            }
        }

        if (maxLayers === 0) continue;

        for (let layerIndex = 0; layerIndex < maxLayers; layerIndex++) {
            const points: PlotPoint[] = [];
            for (const m of metrics) {
                if (m.step > maxStep) continue;
                const arr = m[metric];
                if (Array.isArray(arr) && layerIndex < arr.length && typeof arr[layerIndex] === 'number' && !isNaN(arr[layerIndex])) {
                    points.push({ step: m.step, value: arr[layerIndex] });
                }
            }
            if (points.length > 0) {
                plotData.push([`Layer ${layerIndex}`, points]);
            }
        }

        if (plotData.length === 0) continue;

        const outputFilePath = path.join(outputDir, `plot_${metric}_exp_${expName.replace(/[^a-z0-9]/gi, '_')}.svg`);
        const plotTitle = `${metric} vs. Step for Exp: ${expName}`;
        const yAxisLabel = `${metric} Value (Legend=Layer Index)`;

        createPlot(plotData, outputFilePath, plotTitle, 'Step', yAxisLabel, yMinBound, yMaxBound); // Pass bounds
        console.log(`✅ Experiment plot for "${metric}" Experiment "${expName}" saved to ${outputFilePath}`);
        plotsGenerated++;
    }
    return plotsGenerated;
}


// --- Argument Parsing Helper ---
function parseArgs(args: string[]): { [key: string]: string | boolean } {
    const parsed: { [key: string]: string | boolean } = {};
    for (let i = 0; i < args.length; i++) {
        if (args[i].startsWith('--')) {
            const key = args[i].substring(2);
            const nextVal = args[i + 1];
            if (nextVal && !nextVal.startsWith('--')) {
                parsed[key] = nextVal; // Keep as string for now
                i++;
            } else {
                parsed[key] = true;
            }
        }
    }
    return parsed;
}

// --- Helper to find Max Layer Index ---
function findMaxLayerIndex(experimentData: ExperimentData, metric: ArrayMetric, maxStep: number): number {
    let maxIndex = -1;
    for (const [_, metrics] of experimentData) {
        for (const m of metrics) {
            if (m.step > maxStep) continue;
            const arr = m[metric];
            if (Array.isArray(arr)) {
                maxIndex = Math.max(maxIndex, arr.length - 1);
            }
        }
    }
    return maxIndex;
}


// --- Main Execution Logic ---
const rawArgs = process.argv.slice(2);

// Updated Usage Message
const usage = `❌ Invalid arguments.
Usage:
  ts-node src/plotter.ts <experiment_set_name> <metric|'all'> [options]

Options:
  --mode <layer|experiment>       Required for array metrics (alphaFirstParams, etc.)
  --layer-index <index>           Required for --mode layer
  --experiment-name <name>        Optional for --mode experiment (plots only specified exp)
  --exclude-experiments <n1,n2>   Comma-separated list of experiment names to exclude
  --max-step <N>                  Maximum step value to include in plots
  --y-min <number>                Set minimum Y-axis boundary (final_min = max(y-min, data_min))
  --y-max <number>                Set maximum Y-axis boundary (final_max = min(y-max, data_max))
  --output-dir <dir>              Override output directory for specific plots (not recommended for 'all')

Examples:
  ts-node src/plotter.ts MyRun all --y-min 0 --y-max 1
  ts-node src/plotter.ts MyRun loss --max-step 500 --y-max 1.0
  ts-node src/plotter.ts MyRun alphaFirstParams --mode layer --layer-index 0 --exclude-experiments ExpC,ExpD --y-min 0
  ts-node src/plotter.ts MyRun alphaSecondParams --mode experiment --experiment-name ExpA --max-step 1000`;


if (rawArgs.length < 2) {
    console.error(usage);
    process.exit(1);
}

const experimentSetName = rawArgs[0];
const metricArg = rawArgs[1]; // Can be a specific metric or 'all'
const options = parseArgs(rawArgs.slice(2)); // Parse options like --mode, --layer-index

// --- Path Construction ---
const baseExperimentDir = path.resolve(experimentSetName); // Base directory for the experiment set
const inputJsonPath = path.join(baseExperimentDir, 'metrics.json');

// --- Process Options ---
const excludeExperimentsList: string[] = options['exclude-experiments']
    ? String(options['exclude-experiments']).split(',').map(s => s.trim()).filter(s => s.length > 0)
    : [];

let maxStep: number = Infinity;
if (options['max-step'] !== undefined) {
    const parsedMaxStep = parseInt(String(options['max-step']), 10);
    if (!isNaN(parsedMaxStep) && parsedMaxStep >= 0) {
        maxStep = parsedMaxStep;
    } else {
        console.error(`❌ Invalid --max-step value: "${options['max-step']}". Must be a non-negative integer.`);
        process.exit(1);
    }
}

// Parse Y-axis limit bounds (using Bound suffix now)
let yMinBound: number | undefined = undefined;
if (options['y-min'] !== undefined) {
    const parsedYMin = parseFloat(String(options['y-min']));
    if (!isNaN(parsedYMin)) {
        yMinBound = parsedYMin;
    } else {
        console.error(`❌ Invalid --y-min value: "${options['y-min']}". Must be a number.`);
        process.exit(1);
    }
}
let yMaxBound: number | undefined = undefined;
if (options['y-max'] !== undefined) {
    const parsedYMax = parseFloat(String(options['y-max']));
    if (!isNaN(parsedYMax)) {
        yMaxBound = parsedYMax;
    } else {
        console.error(`❌ Invalid --y-max value: "${options['y-max']}". Must be a number.`);
        process.exit(1);
    }
}

// --- Read and Parse Input Data ---
let rawExperimentData: ExperimentData;
try {
    if (!fs.existsSync(baseExperimentDir) || !fs.lstatSync(baseExperimentDir).isDirectory()) {
        throw new Error(`Experiment directory not found or is not a directory: ${baseExperimentDir}`);
    }
    if (!fs.existsSync(inputJsonPath)) {
        throw new Error(`Input file 'metrics.json' not found inside directory: ${baseExperimentDir}`);
    }
    const jsonData = fs.readFileSync(inputJsonPath, 'utf-8');
    rawExperimentData = JSON.parse(jsonData);
    if (!Array.isArray(rawExperimentData) || rawExperimentData.some(item => !Array.isArray(item) || item.length !== 2 || typeof item[0] !== 'string' || !Array.isArray(item[1]))) {
        throw new Error("Invalid JSON structure in 'metrics.json'. Expected format: [string, Metric[]][]");
    }
} catch (error: any) {
    console.error(`❌ Error reading or parsing JSON file "${inputJsonPath}":`, error.message);
    process.exit(1);
}

// --- Apply Experiment Filter ---
const filteredExperimentData = rawExperimentData.filter(([expName, _]) => !excludeExperimentsList.includes(expName));
if (excludeExperimentsList.length > 0) {
    console.log(`ℹ️ Excluding experiments: ${excludeExperimentsList.join(', ')}`);
}
if (filteredExperimentData.length === 0) {
    console.error("❌ No experiments remaining after applying --exclude-experiments filter. Exiting.");
    process.exit(0); // Not an error, but nothing to do
}


// --- Route to appropriate plotting function ---
try {
    let totalPlotsGenerated = 0;

    // Y-limit bounds are passed globally to all plotting functions called below

    if (metricArg.toLowerCase() === 'all') {
        console.log(`✨ Processing 'all' metrics for experiment set: ${experimentSetName}`);
        console.log(`   (Excluding: ${excludeExperimentsList.length > 0 ? excludeExperimentsList.join(', ') : 'None'}, Max Step: ${maxStep === Infinity ? 'None' : maxStep}, Y-Min Bound: ${yMinBound ?? 'None'}, Y-Max Bound: ${yMaxBound ?? 'None'})`);
        console.warn("⚠️ This may generate a large number of plot files! Y-axis limit bounds apply globally.");

        // --- Plot Scalar Metrics ---
        console.log("\n⏳ Plotting scalar metrics...");
        let scalarPlotsGenerated = 0;
        for (const scalarMetric of SCALAR_METRICS) {
            // Pass global y-limit bounds
            if (plotScalarMetric(filteredExperimentData, scalarMetric, baseExperimentDir, maxStep, yMinBound, yMaxBound)) {
                scalarPlotsGenerated++;
            }
        }
        console.log(`  > Generated ${scalarPlotsGenerated} scalar plot(s).`);
        totalPlotsGenerated += scalarPlotsGenerated;

        // --- Plot Array Metrics (Layer and Experiment) ---
        for (const arrayMetric of ARRAY_METRICS) {
            console.log(`\n⏳ Plotting array metric "${arrayMetric}"...`);

            // Find max layer index based on filtered data and maxStep
            const maxLayerIndex = findMaxLayerIndex(filteredExperimentData, arrayMetric, maxStep);
            if (maxLayerIndex < 0) {
                console.log(`  ℹ️ Metric "${arrayMetric}" not found or has no layer data within filters. Skipping layer/experiment plots.`);
                continue;
            }

            // Define output directories
            const layerPlotsDir = path.join(baseExperimentDir, 'layer_plots');
            const experimentPlotsDir = path.join(baseExperimentDir, 'experiment_plots');

            // Generate Layer Plots
            console.log(`  Generating layer plots (up to index ${maxLayerIndex}) into ${layerPlotsDir}...`);
            let layerPlotsGenerated = 0;
            for (let i = 0; i <= maxLayerIndex; i++) {
                // Pass global y-limit bounds
                if (plotLayerMetric(filteredExperimentData, arrayMetric, i, layerPlotsDir, maxStep, yMinBound, yMaxBound)) {
                    layerPlotsGenerated++;
                }
            }
            console.log(`    > Generated ${layerPlotsGenerated} layer plot(s) for "${arrayMetric}".`);
            totalPlotsGenerated += layerPlotsGenerated;

            // Generate Experiment Plots
            console.log(`  Generating experiment plots into ${experimentPlotsDir}...`);
            // Pass global y-limit bounds
            const experimentPlotsGenerated = plotExperimentMetric(filteredExperimentData, arrayMetric, undefined, experimentPlotsDir, maxStep, yMinBound, yMaxBound);
            console.log(`    > Generated ${experimentPlotsGenerated} experiment plot(s) for "${arrayMetric}".`);
            totalPlotsGenerated += experimentPlotsGenerated;
        }

    } else if (SCALAR_METRICS.includes(metricArg as ScalarMetric)) {
        // Plot specific scalar metric
        const metric = metricArg as ScalarMetric;
        const outputDir = options['output-dir'] as string || baseExperimentDir;
        // Pass global y-limit bounds
        if (plotScalarMetric(filteredExperimentData, metric, outputDir, maxStep, yMinBound, yMaxBound)) {
            totalPlotsGenerated++;
        }

    } else if (ARRAY_METRICS.includes(metricArg as ArrayMetric)) {
        // Plot specific array metric
        const metric = metricArg as ArrayMetric;
        const mode = options['mode'];
        // Pass global y-limit bounds

        if (mode === 'layer') {
            const layerIndex = options['layer-index'];
            const parsedLayerIndex = parseInt(String(layerIndex), 10);
            if (layerIndex === undefined || isNaN(parsedLayerIndex) || parsedLayerIndex < 0) {
                throw new Error("Missing or invalid --layer-index <non-negative integer> for --mode layer.");
            }
            const defaultOutputDir = path.join(baseExperimentDir, 'layer_plots');
            const outputDir = options['output-dir'] as string || defaultOutputDir;
            if (plotLayerMetric(filteredExperimentData, metric, parsedLayerIndex, outputDir, maxStep, yMinBound, yMaxBound)) {
                totalPlotsGenerated++;
            }
        } else if (mode === 'experiment') {
            const targetExpName = options['experiment-name'] as string | undefined;
            if (targetExpName !== undefined && typeof targetExpName !== 'string') {
                throw new Error("Invalid --experiment-name value. Must be a string.");
            }
            const defaultOutputDir = path.join(baseExperimentDir, 'experiment_plots');
            const outputDir = options['output-dir'] as string || defaultOutputDir;
            totalPlotsGenerated += plotExperimentMetric(filteredExperimentData, metric, targetExpName, outputDir, maxStep, yMinBound, yMaxBound);
        } else {
            throw new Error(`Missing or invalid --mode <layer|experiment> for array metric "${metric}".`);
        }
    } else {
        throw new Error(`Invalid metric specified: "${metricArg}". Use 'all', a scalar metric (${SCALAR_METRICS.join(', ')}), or an array metric (${ARRAY_METRICS.join(', ')}) with appropriate options.`);
    }

    console.log(`\n✨ Plotting process completed. ${totalPlotsGenerated} total plot(s) generated.`);

} catch (error: any) {
    console.error(`\n❌ An error occurred during plotting: ${error.message}`);
    console.log(`\n${usage}`); // Show usage on error
    process.exit(1);
}
