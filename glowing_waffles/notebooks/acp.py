class ObserveACP:

    def __init__(self, name, ra=None, dec=None):
        '''
        imports Skycoord, astropy.time, and datetime

        initialize object with given attributes

        Attributes
        ----------
        name: The name of the observed object
        start_time: time of observation as a string in form 00:00:00
        filters: list including the names of filters to be used in observation. case sensitive.
        exposure: exposure time in integer seconds.
        im_count: the number of images taken in each filter as an int
        repeat: how many times the observation is repeated
        binning: the binning of the CCD. should be kept 1.
        ra: the Right Ascension of the object in decimal degrees
        dec: the Declination of the object in decimal degrees

        Parameters
        ----------
        name: the name of the observed object. must be a string
        ra: the right ascension of the aobject
        dec: the declination of the object

        Returns
        -------
        does note return, None
        '''

        from astropy.coordinates import SkyCoord
        from astropy.time import Time
        import datetime

        # set some defaults
        self.name = name
        self.start_time = str(datetime.datetime.today()).split()[1][:8]
        self.filters = []
        self.exposure = 1
        self.im_count = 1
        self.repeat = 1
        self.binning = 1

        if (ra is None) and (dec is None):

            coords = SkyCoord.from_name(self.name)
            self.ra = coords.ra.deg
            self.dec = coords.dec.deg

        elif (ra is None) or (dec is None):
            raise ValueError("Must give both RA and DEC")

        else:

            self.ra = ra
            self.dec = dec

    def wait_until(self, time):
        '''
        sets a time to start observation

        Parameters
        ----------
        time: string representing the time of observation in the form 01:23:45

        Returns
        -------
        Does not return, None
        '''
        self.start_time = time

    def add_filter(self, filt):

        if filt in self.filters:
            print("Filter", filt, "is already in list!")

        elif isinstance(filt, str):
            # do the string thing
            self.filters.append(filt)

        elif not isinstance(filt, str):
            try:
                # loop over elements in filt
                for i in filt:
                    if i in self.filters:
                        print("Filter", i, "is already in list")

                    else:
                        self.filters.append(i)

            except IndexError:
                raise ValueError('filt must be a non-string iterable')

    def rm_filter(self, filt):

        if filt not in self.filters:
            print("No filter", filt, "in list")

        elif isinstance(filt, str):
            self.filters.remove(filt)

        elif not isinstance(filt, str):
            try:
                for i in filt:
                    if i in self.filters:

                        self.filters.remove(i)

                    else:
                        print("No filter", filt, "in list")
            except IndexError:
                raise ValueError('filt must be a non-string iterable')

    def expose(self, num):
        '''
        set an exposure time to be written to .txt file

        Parameters
        ----------
        num: an integer value of the exposure time.

        Returns
        -------
        Does not return, None
        '''
        self.exposure = num

    def count(self, num):
        '''
        sets how many images of each filter you are taking

        Parameters
        ----------
        num: an integer value of the number of images being taken in one filter

        Returns
        -------
        Does not return, None
        '''
        self.im_count = num

    def repeat(self, num):
        '''
        sets a repeat in the .txt file. determines how many times you repeat the observation.

        Parameters
        ----------
        num: an integer value of the number of repeat observations.

        Returns
        -------
        Does not return, None
        '''
        self.repeat = num

    def write(self, title):
        '''
        writes all class attributes to a text file with a given name. All attributes have default values, but are not ideal. please
        go through each method and set values.

        Parameters
        ----------
        title: A string that defines the name of the text file to be written to.

        Returns
        -------
        Does not return, None.

        '''

        scalar = len(self.filters)

        f = open(title, 'w')

        f.write("#waituntil 1, " + str(self.start_time) + '\n')
        f.write("#repeat " + str(self.repeat) + '\n')

        count = scalar * (str(self.im_count) + ',')
        f.write("#count " + count + '\n')

        f.write("#filter ")
        for i in self.filters:
            f.write(i + ',')
        f.write('\n')

        interval = scalar * (str(self.exposure) + ',')
        f.write("#interval " + interval + '\n')

        binning = scalar * (str(self.binning) + ',')
        f.write("#binning " + binning + '\n')

        RA = str(self.ra)
        DEC = str(self.dec)
        f.write(self.name + '    ' + RA + '    ' + DEC + '\n' + '\n')
