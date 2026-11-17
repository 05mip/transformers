# Labeling criteria (one line per type)

Applied when hand-cleaning web images and when labeling our phone photos. An image is kept only if
exactly one type clearly applies and the whole bike (or most of it) is visible. Anything else goes in
`exclude.csv` with a reason.

| Type | Keep if |
|---|---|
| road | Drop (curled-under) handlebars, thin slick tires, lightweight frame; built for pavement speed. |
| mountain | Flat/riser bars, wide knobby tires, usually front or full suspension. |
| cruiser | Swept-back upright bars, balloon tires, long curved/relaxed frame (beach-cruiser style). |
| bmx | Small (~20") wheels, single gear, compact low frame, high-rise bars. |
| tricycle | Human-powered, three wheels, one rider (child or adult upright). No motors, no cargo/rickshaw bodies. |
| folding | Visible fold hinge(s) or shown folded; small wheels typical. |
| hybrid | Flat bars and upright position but medium-width, mostly smooth tires and no/light suspension (between road and mountain). |
| tandem | Two (or more) saddles and pedal sets in line on one frame, two wheels. |

**Reject reasons** (use these words in `exclude.csv`): `wrong_type`, `ambiguous` (two types fit),
`not_photo` (drawing, diagram, ad), `no_bike_visible` (bike tiny, cropped or only a part),
`motorized`, `duplicate`, `other`.
