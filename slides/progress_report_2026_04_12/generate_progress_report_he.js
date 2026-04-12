"use strict";

const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");
const { calcTextBox } = require("./pptxgenjs_helpers/text");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers/layout");
const { safeOuterShadow } = require("./pptxgenjs_helpers/util");

const OUTPUT_NAME = "progress_report_he_2026_04_12.pptx";
const outputPath = path.join(__dirname, OUTPUT_NAME);

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "OpenAI Codex";
pptx.company = "SLT_2026";
pptx.subject = "Depth Anything V2 progress report";
pptx.title = "דו\"ח התקדמות - Depth Anything V2";
pptx.lang = "he-IL";
pptx.theme = {
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  lang: "he-IL",
};

const colors = {
  paper: "F7FAFC",
  white: "FFFFFF",
  ink: "10233D",
  inkSoft: "55697F",
  border: "D7E1EA",
  accent: "0F7B6C",
  accentSoft: "D8F0EB",
  warning: "F28C28",
  warningSoft: "FFF2E3",
  info: "3963D3",
  infoSoft: "E8EEFF",
  danger: "B94A48",
  dangerSoft: "FBE8E8",
  panel: "19324A",
  line: "E6EDF4",
  muted: "EEF3F7",
};

const page = { w: 13.333, h: 7.5 };

function addBase(slide) {
  slide.background = { color: colors.paper };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: 0.32,
    h: page.h,
    line: { color: colors.panel, transparency: 100 },
    fill: { color: colors.panel },
  });
}

function addMeasuredText(slide, text, x, y, w, fontSize, options = {}) {
  const layout = calcTextBox(fontSize, {
    text,
    w,
    fontFace: "Arial",
    margin: 0,
    breakLine: options.breakLine ?? false,
  });

  const h = Math.max(layout.h, options.minHeight || 0.18);
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: "Arial",
    fontSize,
    color: options.color || colors.ink,
    bold: options.bold || false,
    align: options.align || "right",
    valign: options.valign || "mid",
    margin: 0,
    rtlMode: true,
    breakLine: options.breakLine ?? false,
    paraSpaceAfterPt: options.paraSpaceAfterPt ?? 0,
    lineSpacingMultiple: options.lineSpacingMultiple || 1.05,
  });
  return h;
}

function addBulletList(slide, items, x, y, w, fontSize, options = {}) {
  const runs = [];
  items.forEach((item, index) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: 14 },
        hanging: 3,
      },
    });
    if (index < items.length - 1) runs.push({ text: "\n" });
  });

  const layout = calcTextBox(fontSize, {
    text: items.join("\n"),
    w,
    fontFace: "Arial",
    margin: 0,
    breakLine: true,
  });

  slide.addText(runs, {
    x,
    y,
    w,
    h: layout.h + 0.08,
    fontFace: "Arial",
    fontSize,
    color: options.color || colors.ink,
    margin: 0,
    valign: "top",
    rtlMode: true,
    paraSpaceAfterPt: options.paraSpaceAfterPt ?? 7,
    breakLine: true,
  });
  return layout.h + 0.08;
}

function addTitle(slide, title, subtitle) {
  addMeasuredText(slide, title, 0.88, 0.55, 11.65, 24, {
    bold: true,
  });
  addMeasuredText(slide, subtitle, 0.88, 1.16, 11.65, 11, {
    color: colors.inkSoft,
  });
}

function addPanel(slide, x, y, w, h, fill = colors.white) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.1,
    fill: { color: fill },
    line: { color: colors.border, pt: 1.1 },
    shadow: safeOuterShadow("7F8E9E", 0.1, 45, 1.2, 0.6),
  });
}

function addPill(slide, text, x, y, w, fill, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.34,
    rectRadius: 0.05,
    fill: { color: fill },
    line: { color: fill, transparency: 100 },
  });
  addMeasuredText(slide, text, x, y + 0.11, w, 9.5, {
    align: "center",
    bold: true,
    color,
  });
}

function addLabelText(slide, text, x, y, w, color) {
  addMeasuredText(slide, text, x, y, w, 10.5, {
    bold: true,
    color,
  });
}

function addFooter(slide, text) {
  slide.addShape(pptx.ShapeType.line, {
    x: 0.88,
    y: 6.88,
    w: 11.55,
    h: 0,
    line: { color: colors.line, pt: 1 },
  });
  addMeasuredText(slide, text, 0.88, 6.97, 11.55, 9, {
    color: colors.inkSoft,
  });
}

function addProgressBar(slide, label, percentText, ratio, x, y, w, fill) {
  addMeasuredText(slide, label, x, y, w - 0.75, 12, {
    bold: true,
  });
  addMeasuredText(slide, percentText, x + w - 0.55, y, 0.55, 11, {
    align: "left",
    bold: true,
    color: colors.inkSoft,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y: y + 0.38,
    w,
    h: 0.12,
    rectRadius: 0.04,
    fill: { color: colors.muted },
    line: { color: colors.muted, transparency: 100 },
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y: y + 0.38,
    w: w * ratio,
    h: 0.12,
    rectRadius: 0.04,
    fill: { color: fill },
    line: { color: fill, transparency: 100 },
  });
}

function finalizeSlide(slide) {
  warnIfSlideHasOverlaps(slide, pptx, {
    muteContainment: true,
    ignoreDecorativeShapes: true,
  });
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

function buildCoverSlide() {
  const slide = pptx.addSlide();
  addBase(slide);

  addPanel(slide, 0.92, 1.18, 5.8, 4.95);
  addPanel(slide, 7.02, 1.18, 5.28, 4.95, colors.panel);

  addLabelText(slide, "דו\"ח התקדמות", 1.24, 1.46, 2.1, colors.accent);
  addMeasuredText(slide, "Depth Anything V2", 1.24, 2.03, 5.05, 26, {
    bold: true,
  });
  addMeasuredText(slide, "עדכון מצב נכון ל־12.04.2026", 1.24, 2.84, 5.05, 15, {
    color: colors.inkSoft,
  });
  addMeasuredText(
    slide,
    "המטרה בשלב הזה היא להראות מה באמת בוצע עד עכשיו: הקמת הסביבה, הכנת הדאטה, מצב ה־benchmarks והחסמים לפני שלב הוולידציה.",
    1.24,
    3.28,
    4.95,
    13,
    { color: colors.inkSoft, breakLine: true, valign: "top" }
  );
  addMeasuredText(slide, "מגישים", 1.24, 5.08, 4.95, 12, {
    bold: true,
    color: colors.inkSoft,
  });
  addMeasuredText(
    slide,
    1.24,
    5.6,
    4.95,
    14,
    { breakLine: true, valign: "top" }
  );

  addMeasuredText(slide, "תמונת מצב", 7.38, 1.56, 4.5, 18, {
    bold: true,
    color: colors.white,
  });
  addProgressBar(slide, "סביבת עבודה והפעלה ראשונית", "100%", 1, 7.38, 2.26, 4.2, colors.accent);
  addProgressBar(slide, "רכישת דאטה ו־preprocessing", "72%", 0.72, 7.38, 3.2, 4.2, colors.warning);
  addProgressBar(slide, "מוכנות לוולידציה", "45%", 0.45, 7.38, 4.14, 4.2, colors.info);
  addMeasuredText(
    slide,
    "הסביבה עובדת, DA-2K מוכן, ושני benchmarks נוספים כבר ירדו למכונה.",
    7.38,
    5.22,
    4.18,
    12,
    { color: "D8E6F2", breakLine: true, valign: "top" }
  );

  addFooter(slide, "SLT_2026 | דו\"ח התקדמות מבוסס על העבודה שבוצעה בפועל במאגר");
  finalizeSlide(slide);
}

function buildCompletedSlide() {
  const slide = pptx.addSlide();
  addBase(slide);
  addTitle(
    slide,
    "מה כבר הושלם",
    "החלקים שבהם אפשר כבר להצביע על תוצאה עובדת ולא רק על תכנון"
  );

  addPanel(slide, 0.92, 1.8, 7.2, 4.7);
  addLabelText(slide, "עבודה שבוצעה", 1.18, 1.94, 2.0, colors.accent);
  addBulletList(
    slide,
    [
      "סודר מבנה פרויקט נקי עם תיקיית data ייעודית למודלים, דאטה, פלטים וסקריפטים.",
      "חובר המאגר למימוש הרשמי של Depth Anything V2 ונקבע commit קבוע לשחזור יציב.",
      "הוקמה סביבת Python מקומית והותקנו התלויות הנדרשות.",
      "הורד checkpoint קטן והרצנו sanity check מלא על תמונות הדוגמה של הפרויקט.",
      "נוצרו 20 פלטים תחת data/outputs/sanity_check, ולכן הוכחנו שהקוד עובד מקומית.",
      "הורדנו את DA-2K, חילצנו את הדאטה, ואימתנו את מבנה ההערות.",
      "נבנו manifest וסטטיסטיקות, וטופלה גם אי־התאמה אחת בשם קובץ בתוך הדאטה."
    ],
    1.18,
    2.3,
    6.65,
    12
  );

  addPanel(slide, 8.42, 1.8, 3.88, 2.0);
  addLabelText(slide, "מספרים", 8.7, 1.94, 1.4, colors.info);
  addMeasuredText(slide, "1033", 8.72, 2.28, 1.45, 26, {
    bold: true,
    color: colors.info,
    align: "center",
  });
  addMeasuredText(slide, "תמונות ב־DA-2K", 8.65, 3.1, 1.6, 11, {
    align: "center",
    color: colors.inkSoft,
  });
  addMeasuredText(slide, "2068", 10.46, 2.28, 1.45, 26, {
    bold: true,
    color: colors.accent,
    align: "center",
  });
  addMeasuredText(slide, "זוגות השוואה", 10.36, 3.1, 1.65, 11, {
    align: "center",
    color: colors.inkSoft,
  });

  addPanel(slide, 8.42, 4.1, 3.88, 2.4);
  addLabelText(slide, "משמעות", 8.7, 4.24, 1.4, colors.warning);
  addMeasuredText(
    slide,
    "הגענו לנקודה שבה שלב ההקמה כבר סגור, והמאמץ המרכזי עובר מהקמת סביבה להכנת benchmarks ולוולידציה.",
    8.7,
    4.66,
    3.3,
    12,
    { color: colors.inkSoft, breakLine: true, valign: "top" }
  );

  addFooter(slide, "סיכום השלב: setup מלא + sanity check תקין + DA-2K מוכן להמשך");
  finalizeSlide(slide);
}

function buildBenchmarksSlide() {
  const slide = pptx.addSlide();
  addBase(slide);
  addTitle(
    slide,
    "מצב ה־benchmarks",
    "מה כבר מוכן, מה הורד מקומית, ומה עדיין ממתין לפני הוולידציה המלאה"
  );

  addPanel(slide, 0.92, 1.8, 11.38, 4.7);

  const headers = [
    { label: "Benchmark", x: 1.14, w: 2.15 },
    { label: "סטטוס", x: 3.46, w: 1.5 },
    { label: "מה בוצע", x: 5.18, w: 2.45 },
    { label: "הערות", x: 7.85, w: 4.05 },
  ];

  headers.forEach((header) => {
    addMeasuredText(slide, header.label, header.x, 1.98, header.w, 10.5, {
      bold: true,
    });
  });

  const rows = [
    ["DA-2K", "מוכן", "הורדה, חילוץ ו־preprocessing מלא", "1033 תמונות ו־2068 זוגות. זהו היעד הראשון להרצה."],
    ["KITTI", "הורד", "הקובץ נשמר מקומית", "עדיין נדרש extraction והכנת pipeline להערכה."],
    ["NYU-Depth-v2", "הורד", "קובץ MAT נשמר מקומית", "הדאטה קיים אך טרם עבר preprocessing."],
    ["Sintel", "חסום", "הורדה נעצרה", "אין כרגע מספיק מקום פנוי בדיסק להשלמת התהליך."],
    ["ETH3D / DIODE", "ממתין", "הסקריפט כבר מוכן", "ההורדה תתבצע אחרי פינוי מקום פנוי."],
  ];

  let y = 2.34;
  rows.forEach((row, index) => {
    const rowFill = index % 2 === 0 ? "FAFCFE" : "F3F7FB";
    slide.addShape(pptx.ShapeType.rect, {
      x: 1.08,
      y,
      w: 10.92,
      h: 0.72,
      line: { color: rowFill, transparency: 100 },
      fill: { color: rowFill },
    });

    addMeasuredText(slide, row[0], 1.14, y + 0.22, 2.1, 10.5, { bold: true });

    const statusColor =
      row[1] === "מוכן"
        ? colors.accent
        : row[1] === "הורד"
        ? colors.warning
        : row[1] === "חסום"
        ? colors.danger
        : colors.info;

    addMeasuredText(slide, row[1], 3.58, y + 0.22, 1.08, 10.2, {
      bold: true,
      color: statusColor,
    });
    addMeasuredText(slide, row[2], 5.18, y + 0.12, 2.45, 9.6, {
      color: colors.inkSoft,
      breakLine: true,
      valign: "top",
    });
    addMeasuredText(slide, row[3], 7.85, y + 0.12, 4.05, 9.6, {
      color: colors.inkSoft,
      breakLine: true,
      valign: "top",
    });

    y += 0.76;
  });

  addFooter(slide, "סטטוס מעשי: benchmark אחד מוכן, שניים כבר זמינים מקומית, והשאר ממתינים רק לתנאי דיסק");
  finalizeSlide(slide);
}

function buildBlockersSlide() {
  const slide = pptx.addSlide();
  addBase(slide);
  addTitle(
    slide,
    "חסמים והצעד הבא",
    "מה מעכב אותנו כרגע, ומה צריך לעשות כדי לעבור להרצת benchmark ראשון"
  );

  addPanel(slide, 0.92, 1.8, 5.4, 4.7);
  addLabelText(slide, "החסם המרכזי", 1.18, 1.94, 2.0, colors.danger);
  addBulletList(
    slide,
    [
      "במהלך העבודה נשארו בערך 4.1GiB פנויים בלבד על הדיסק.",
      "הורדות גדולות כמו Sintel, ETH3D ו־DIODE אינן בטוחות בתנאי האחסון הנוכחיים.",
      "ההחלטה הייתה לעצור מוקדם כדי לא לייצר קבצים חלקיים ומצב שקשה לשחזר.",
      "שיפרנו את סקריפט ההורדה כך שהוא בודק מקום פנוי מראש ונכשל מוקדם עם הודעה ברורה."
    ],
    1.18,
    2.3,
    4.88,
    12
  );

  addPanel(slide, 6.62, 1.8, 5.68, 4.7);
  addLabelText(slide, "הפעולות הבאות", 6.88, 1.94, 2.1, colors.accent);
  addBulletList(
    slide,
    [
      "לפנות מקום בדיסק ולהשלים את ההורדה של Sintel, ETH3D ו־DIODE.",
      "לבצע extraction ו־preprocessing גם עבור KITTI ו־NYU-Depth-v2.",
      "לממש או לאמת את metrics כך שהתוצאות יהיו ברות השוואה למאמר.",
      "להריץ benchmark ראשון על DA-2K ולהפיק טבלת תוצאות ראשונה."
    ],
    6.88,
    2.3,
    5.02,
    12
  );

  addMeasuredText(
    slide,
    "יעד לשקופית ההתקדמות הבאה: תוצאה ראשונה של benchmark עם metric מחושב והשוואה לערכי המאמר.",
    6.88,
    5.46,
    5.0,
    10.6,
    { color: colors.panel, bold: true, breakLine: true, valign: "top" }
  );

  addFooter(slide, "מכאן עוברים מעבודת הכנה טכנית לעבודה ניסויית שאפשר למדוד ולהציג");
  finalizeSlide(slide);
}

async function main() {
  fs.mkdirSync(__dirname, { recursive: true });
  buildCoverSlide();
  buildCompletedSlide();
  buildBenchmarksSlide();
  buildBlockersSlide();
  await pptx.writeFile({ fileName: outputPath });
  console.log(`Wrote ${outputPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
