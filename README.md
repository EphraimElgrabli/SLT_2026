# שחזור Depth Anything V2


המאגר מרכז את עבודת השחזור למאמר **Depth Anything V2**. מטרת הפרויקט היא לבדוק עד כמה ניתן לשחזר את תוצאות המאמר בעזרת הקוד וה־checkpoints הציבוריים, בלי לשחזר את תהליך האימון הפרטי של החוקרים.

העבודה מחולקת לשני צירי הערכה:

- **DA-2K**: שחזור relative depth באמצעות זוגות נקודות מסומנות.
- **Pixelwise / Metric depth benchmarks**: הערכה על KITTI, NYU Depth V2, Sintel, DIODE ו־ETH3D בעזרת מדדים כמו `AbsRel`, `RMSE` ו־`delta1`.

## סטטוס נוכחי

שלב `DA-2K` הושלם במלואו: הדאטה הורד, עבר preprocessing, נבנה pipeline הערכה, הורצו שלושת מודלי הסטודנט הציבוריים, ונוצר דו"ח שחזור עם hashes לתוצרים.

תוצאות `DA-2K` משחזרות את טבלת המאמר בצורה קרובה מאוד:

| מודל | תוצאה בפרויקט | תוצאה במאמר |
| --- | ---: | ---: |
| `ViT-S` | `95.16%` | `95.30%` |
| `ViT-B` | `97.05%` | `97.00%` |
| `ViT-L` | `97.10%` | `97.10%` |

בנוסף, קיים pipeline להערכת עומק מטרי על `KITTI`, `NYU Depth V2`, `Sintel`, `DIODE` ו־`ETH3D`. חשוב לשים לב: עבור `ETH3D` ההערכה הנוכחית מבוססת על תצפיות COLMAP דלילות ולא על שחזור dense scan-depth מלא כמו בפרוטוקול המחברים. לכן יש לפרש את התוצאות שם כהערכה מוגבלת פרוטוקול, לא כהשוואה מלאה וזהה למאמר.

## מבנה הפרויקט

```text
.
├── README.md
├── data/
│   ├── README.md
│   ├── datasets/        # דאטה מקומי, לא מיועד לניהול מלא בגיט
│   ├── external/        # מימושים חיצוניים משוכפלים, כולל Depth Anything V2
│   ├── models/          # checkpoints ציבוריים
│   ├── outputs/         # תחזיות, מדדים ופלטי sanity check
│   └── scripts/         # סקריפטי workflow מרכזיים
├── paper/               # המאמר וחומרי מקור
├── reports/             # דוחות שחזור ותוצרי הגשה
├── slides/              # מצגות התקדמות
├── tests/               # בדיקות מהירות
└── writing/             # הערות עבודה, ראיות ותכנון הדוח
```

נקודת הכניסה המרכזית היא:

```bash
python data/scripts/slt_data.py --help
```

הבחירה בארכיטקטורה הזו שומרת על הפרדה ברורה בין קוד workflow, דאטה מקומי, תוצרים, דוחות, ומסמכי עבודה. כך ניתן להריץ מחדש חלקים מהשחזור בלי לערבב בין קוד, קבצים כבדים ופלטים שנוצרים מהרצות.

## סביבת עבודה מומלצת

- **שפה וריצה:** Python 3.
- **ניהול סביבה:** virtual environment מקומי תחת `data/.venv`, שנוצר דרך `setup`.
- **מימוש חיצוני:** המאגר הרשמי של `Depth Anything V2`, משוכפל לתוך `data/external` בגרסה מקובעת.
- **מודלים:** checkpoints ציבוריים של Depth Anything V2.
- **בדיקות:** `unittest` עבור שכבת ה־CLI והעברת ארגומנטים.
- **תוצרים:** JSON/JSONL למדדים ותחזיות, ו־DOCX לדוחות שחזור.

אין בפרויקט בסיס נתונים, שירות backend, או תשתית deployment. זהו פרויקט שחזור מחקרי מקומי, ולכן הדגש הוא על reproducibility, הפרדת תוצרים, ותיעוד מגבלות הפרוטוקול.

## הרצה ראשונית

להקמת סביבת העבודה והרצת smoke test:

```bash
python data/scripts/slt_data.py setup
```

הפקודה:

- משכפלת את המימוש הרשמי של `Depth Anything V2`.
- מקימה סביבת Python מקומית.
- מתקינה את התלויות הנדרשות.
- מורידה checkpoints ציבוריים.
- מריצה sanity check על תמונות הדוגמה.

## שחזור DA-2K

```bash
python data/scripts/slt_data.py da2k
python data/scripts/slt_data.py evaluate-da2k
python data/scripts/python/build_da2k_report.py
```

התוצרים המרכזיים:

- `data/outputs/da2k/predictions/*.jsonl` - תחזיות לכל זוג נקודות ולכל מודל.
- `data/outputs/da2k/summary.json` - accuracy כללי ולפי סוג סצנה.
- `reports/da2k_reproduction.docx` - דו"ח שחזור מסודר.
- `reports/da2k_artifact_hashes.txt` - hashes של התוצרים המרכזיים.

בשלב זה עובדו `1033` תמונות ו־`2068` זוגות נקודות מתוך `DA-2K`.

## הערכות Pixelwise ו־Metric Depth

להורדת checkpoints מטריים:

```bash
python data/scripts/slt_data.py metric-checkpoints
```

להרצת הערכה מטרית על הדאטה המקומי:

```bash
python data/scripts/slt_data.py evaluate-metric --datasets kitti nyu_depth_v2 sintel diode eth3d
```

התוצרים נכתבים אל:

- `data/outputs/metric_depth/summary.json`
- `data/outputs/metric_depth/*_details.json`
- `data/outputs/metric_depth/*_records.jsonl`

קיימים גם workflows ייעודיים להכנה או הורדה של benchmarks מסוימים:

```bash
python data/scripts/slt_data.py acquire-kitti
python data/scripts/slt_data.py prepare-kitti
python data/scripts/slt_data.py acquire-nyu
python data/scripts/slt_data.py prepare-nyu
python data/scripts/slt_data.py prepare-sintel
python data/scripts/slt_data.py prepare-diode
python data/scripts/slt_data.py prepare-eth3d
```

להפקת דו"ח pixelwise:

```bash
python data/scripts/slt_data.py evaluate-pixelwise
python data/scripts/slt_data.py build-pixelwise-report
```

## בדיקות

הבדיקות המהירות אינן דורשות דאטה או checkpoints:

```bash
python -m unittest discover -s tests -v
```

מומלץ להריץ אותן לפני כל שינוי ב־CLI או בסקריפטי workflow. אם מוסיפים adapter חדש לדאטה או משנים מדדים, כדאי להוסיף בדיקות יחידה למדדים, parsing של manifests, והתנהגות שגיאה כאשר קבצי מקור חסרים.

## מגבלות ידועות

- לא שוחזר תהליך האימון המלא של Depth Anything V2. קוד האימון הפרטי, מימושי loss מדויקים, ו־pipeline של כ־62 מיליון pseudo labels אינם זמינים לציבור.
- `ETH3D` אינו נבדק כרגע מול dense scan-depth מלא. ההערכה הנוכחית מבוססת על תצפיות sparse מתוך קבצי COLMAP.
- ב־`DIODE`, תת־הקבוצה indoor קרובה יותר למאמר, בעוד שתת־הקבוצה outdoor סוטה משמעותית. כנראה קיימים הבדלי valid-mask או טיפול ב־ground truth שלא מתועדים במלואם במאמר.
- חלק מהדאטה והמודלים כבדים ואינם מתאימים לניהול רגיל בגיט. יש לשמור אותם תחת `data/datasets`, `data/models` ו־`data/outputs` לפי מבנה הפרויקט.

## עקרונות תחזוקה

- לשמור את `data/scripts/slt_data.py` כנקודת הכניסה המרכזית, ולא לפזר הוראות הרצה בין סקריפטים לא מתועדים.
- להפריד בין acquisition, preprocessing, evaluation ו־report generation.
- לא להכניס לגיט קבצי דאטה כבדים, checkpoints, caches או פלטים זמניים.
- לתעד שינויי פרוטוקול בדוחות או בקובצי README סמוכים.
- להוסיף בדיקות כאשר משנים CLI, מדדים, parsing או כללי evaluation.

## המשך עבודה מומלץ

1. להשלים בדיקות יחידה עמוקות יותר עבור metric adapters ומדדי עומק.
2. לשפר את תיעוד הפרוטוקול עבור `DIODE` ו־`ETH3D`, במיוחד סביב masks, caps ו־ground-truth projection.
3. להוסיף script מסודר שמוודא אילו תוצרים קיימים מקומית ואילו חסרים לפני הרצה מלאה.
4. להוסיף workflow CI קל שמריץ lint בסיסי ובדיקות מהירות שאינן דורשות דאטה.
5. לשמור את הדוחות והמצגות מסונכרנים עם פקודות ההרצה והתוצרים בפועל.
