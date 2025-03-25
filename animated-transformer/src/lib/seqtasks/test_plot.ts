import * as Plot from '@observablehq/plot';
import { JSDOM } from "jsdom";
import sharp from "sharp";
import fs from "fs";
import { Dictionary } from 'underscore';

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
    marks: [Plot.lineY(b, { x: "step", y: "loss", stroke: "name" }), Plot.text(b, Plot.selectLast(
        { x: "step", y: "loss", z: "name", text: "name", textAnchor: "start", dx: 3 }))
    ],
    width: 500,
    height: 500,
    x: {
        label: "Epoch", // X-axis label
        labelAnchor: "center",
    },
    y: {
        label: "Loss", // Y-axis labes
        labelAnchor: "center",
    },
});

let svgString = plot.outerHTML;
// const svgString: string = plot.outerHTML.replace(
//     "<svg",
//     `<svg><rect width="100%" height="100%" fill="white"/>`
// );

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