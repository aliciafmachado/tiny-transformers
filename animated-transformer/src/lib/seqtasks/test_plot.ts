import * as Plot from '@observablehq/plot';
import { JSDOM } from "jsdom";
import sharp from "sharp";
import fs from "fs";

interface Metric {
    loss: number;
    step: number;
}

let losses: number[] = [0.6, 0.4, 0.2];
let losses2: number[] = [0.3, 0.2, 0.22];

let metrics: Metric[] = losses.map((value, idx) => ({ "loss": value, "step": idx }) as Metric);
let metrics2: Metric[] = losses2.map((value, idx) => ({ "loss": value, "step": idx }) as Metric);

let c: Metric = { "loss": 1, "step": 0.2 }

let a: [string, Metric[]][] = [
    ["loss1", metrics],
    ["loss2", metrics2],
];

let b = a.flatMap(([name, values]) => values.flatMap(d => ({ name, ...d })));

let plot = Plot.plot({
    document: new JSDOM("").window.document,
    marks: [
        Plot.frame(),
        Plot.lineY(b, { x: "step", y: "loss", stroke: "name" }),
        Plot.text(b, Plot.selectLast(
            { x: "step", y: "loss", z: "name", text: "name", textAnchor: "start", dx: 3 })),
        Plot.text(['Loss computation.'], { frameAnchor: "top", dy: -30 }),
    ],
    color: { legend: true },
    figure: false,
    x: {
        label: "Step", // X-axis label
        labelAnchor: "center",
        // n epochs.
        domain: [0, 2],
    },
    y: {
        label: "Loss", // Y-axis labes
        labelAnchor: "center",
        domain: [0, 1],
    },
    // This does not do anything...
    // style: {
    //     background: 'white'
    // },
    marginTop: 40,
    marginRight: 40,
});

// TODO(@aliciafmachado): Not sure how to merge the legend plot into the other plot.
// const legend = Plot.legend({ color: { type: "linear" } });

// TODO(@aliciafmachado): Hacky way to pass the background.
const svgString: string = plot.outerHTML.replace(
    "<style>",
    `<rect width="100%" height="100%" fill="white"/><style>`
);

async function callSharp(): Promise<void> {
    try {
        const buffer: Buffer = await sharp(Buffer.from(svgString, "utf-8")).png().toBuffer();
        fs.writeFileSync("output.png", buffer);
        console.log("JPEG image saved as output.jpeg");
    } catch (error) {
        console.error("Error saving JPEG:", error);
    }
}

callSharp();