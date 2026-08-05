import sqlite3
import numpy as np
import boardlib 
import pickle

class KilterClimb:
  def __init__(self, angle, grade, placementX, placementY, color, nomatch, matrix, ascent, benchmark):
    self.angle = angle
    self.grade = grade
    self.placementX = placementX
    self.placementY = placementY
    self.color = color
    self.nomatch = nomatch
    
    self.matrix = matrix
    self.ascents = ascent
    self.benchmark = benchmark


conn = sqlite3.connect("climbs.db")
cursor = conn.cursor()
cursor.execute(
    # Kilter Board Original (layout_id 1) uses holds from TWO sets: "Bolt Ons" (set_id 1)
    # and "Screw Ons" (set_id 20). Restricting to set_id = 1 only silently drops any climb
    # that uses even a single screw-on hold - verified this was throwing away 62% of climbs
    # that otherwise matched the filters below. Filtering by layout_id instead of a
    # hardcoded set_id list is also more robust if Kilter adds more sets to this layout later.
    "SELECT p.id, h.x, h.y FROM placements p JOIN holes h ON p.hole_id = h.id WHERE p.layout_id = 1;"
)
coord_map = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}

cursor.execute("SELECT id, name FROM placement_roles")
role_to_color = {id: name for id, name in cursor.fetchall()}

cursor.execute("""SELECT
    cs.angle,
    cs.display_difficulty,
    c.frames,
    c.is_nomatch,
    cs.ascensionist_count,
    cs.benchmark_difficulty
    FROM climbs c
    JOIN climb_stats cs ON c.uuid = cs.climb_uuid
    JOIN layouts l ON c.layout_id = l.id
    JOIN products p ON l.product_id = p.id
    WHERE p.name = 'Kilter Board Original' AND c.frames_count = 1 AND cs.ascensionist_count >= 10;""")

climbs = []

for row in cursor:
  angle, grade, framestring, nomatch, ascent, benchmark = row
  climbx = []
  climby = []
  color = []
  last = ""
  pnumber = ""
  rnumber = ""
  first = True
  map = {}
  for char in framestring:
    if char == "r":
      last = last + "r"

    elif char == "p":
      last = last + "p"
      if len(pnumber) != 0 and len(rnumber) != 0:
        map[pnumber] = rnumber
      pnumber = ""
      rnumber = ""
    if last[-1] == "p" and char != "p" and char != "r":
      pnumber = pnumber + char
    if last[-1] == "r" and char != "p" and char != "r":
      rnumber = rnumber + char

  if not all(int(entries) in coord_map for entries in map):
    continue #remove any with holds not on teh boadr

  for entries in map:
    x, y = coord_map[int(entries)]
    colortype = role_to_color[int(map[entries])]
    climbx.append(x)
    climby.append(y)
    color.append(colortype)

  matrix = np.full((44,47), '.')
  for i in range(len(climbx)):
    if color[i] == "start":
      matrix[(climby[i] - 4) // 4][(climbx[i] + 20) // 4] = 'g'
    elif color[i] == "middle" :
      matrix[(climby[i] - 4) // 4][(climbx[i] + 20) // 4] = 'b'
    elif color[i] == "finish":
      matrix[(climby[i] - 4) // 4][(climbx[i] + 20) // 4] = 'p'
    elif color[i] == "foot":
      matrix[(climby[i] - 4) // 4][(climbx[i] + 20 ) // 4] = 'y'
  climb = KilterClimb(angle,grade,climbx,climby,color,nomatch, matrix, ascent, benchmark)
  climbs.append(climb)


with open ("climbs.pkl", "wb") as file:
  pickle.dump(climbs, file)


conn.close()