import alignment
panorama = dict(('path', 'resources/campus/panorama'),
		('model', alignment.eTranslate),
		('percentTopMatches', 20),
		('numberOfRANSAC', 500),
		('RANSACThreshold', 5.0),
		('blendWidth', 50),
		('360degree', 1),
		('focalLength', 595),
		('k1', -0.15),
		('k2', 0.0),
		('filename', 'campus_translation.jpg'))

